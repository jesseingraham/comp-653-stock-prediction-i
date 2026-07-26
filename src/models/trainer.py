"""Standardized training and hyperparameter tuning for the forecasting RNNs.

Provides one pipeline so the price-only and pattern-augmented runs of
`RNNForecaster` are tuned and evaluated identically -- the only intended
difference is the feature set each is given. Robustness comes from expanding-
window walk-forward cross-validation: every tuning trial trains one model per
fold in lockstep and is scored on the mean validation RMSE across folds.

Public pieces:
    * `walk_forward_splits` -- expanding-window CV fold indices.
    * `rmse` / `directional_accuracy` -- evaluation metrics.
    * `train_folds` -- generator training one model per fold in lockstep for a
      single config, yielding mean validation RMSE per epoch (what ASHA prunes
      on). Checkpointable, so a Population Based Training scheduler can be added
      later without changing the training contract.
    * `tune_model` -- Optuna (TPE) search + ASHA early stopping over a shared
      space, results persisted to the Shared Drive and resumable.
    * `evaluate_on_test` -- retrain the best config on all development data and
      report test-set metrics.

Ray is imported lazily inside `tune_model` so the rest of the module (splits,
metrics, training loop) can be used and tested without Ray installed. The module
depends only on the ``(n, seq_len, num_features)`` tensor contract, not on the
feature/windowing code, so it can be built ahead of that stage.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from config import DRIVE_MODELS_PATH
from src.models.rnn_forecaster import RNNForecaster

# Metrics live in src.utils.metrics (the eval domain) and are re-exported here
# so existing callers keep working.
from src.utils.metrics import (  # noqa: E402
    directional_accuracy,
    rmse,
    summarize_forecast,
)

__all__ = [
    "directional_accuracy",
    "evaluate_on_test",
    "rmse",
    "train_folds",
    "tune_model",
    "walk_forward_splits",
]

# A scaler factory returns a fresh, unfitted sklearn-style scaler (fit /
# transform) to be fitted per fold on training data only.
ScalerFactory = Callable[[], Any]


def set_seed(seed: int) -> None:
    """Seed NumPy and torch RNGs for reproducible runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def walk_forward_splits(
    n_samples: int,
    n_folds: int = 4,
    val_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build expanding-window walk-forward cross-validation folds.

    Validation blocks tile the end of the (chronologically ordered) sequence;
    each fold's training set is everything strictly before its validation block,
    so the training window expands fold to fold and never sees the future.

    Args:
        n_samples: Number of chronologically ordered samples.
        n_folds: Number of expanding folds.
        val_size: Samples per validation block. Defaults to
            ``n_samples // (n_folds + 1)``.

    Returns:
        A list of ``(train_idx, val_idx)`` integer-array pairs, earliest first.

    Raises:
        ValueError: If the requested folds do not leave room for training data.
    """
    if val_size is None:
        val_size = n_samples // (n_folds + 1)
    if val_size < 1:
        raise ValueError("val_size resolves to < 1; too many folds for data.")

    first_val_start = n_samples - n_folds * val_size
    if first_val_start < 1:
        raise ValueError(
            f"{n_folds} folds of {val_size} leave no training data "
            f"(need < {n_samples} total validation samples)."
        )

    indices = np.arange(n_samples)
    splits = []
    for k in range(n_folds):
        val_start = first_val_start + k * val_size
        val_end = val_start + val_size
        splits.append((indices[:val_start], indices[val_start:val_end]))
    return splits


def _to_tensor(array: Any) -> torch.Tensor:
    """Convert a NumPy array or tensor to a float32 tensor."""
    return torch.as_tensor(np.asarray(array), dtype=torch.float32)


def _scale_features(
    scaler: Any | None,
    x_train: np.ndarray,
    x_other: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a scaler on the training windows and apply it to both sets.

    Fitting on `x_train` only keeps each fold leakage-free. A ``None`` scaler is
    a no-op.

    Args:
        scaler: A fresh sklearn-style scaler, or ``None``.
        x_train: Training windows, shape ``(n, seq_len, num_features)``.
        x_other: Windows to transform with the fitted scaler (val or test).

    Returns:
        The scaled ``(x_train, x_other)`` arrays.
    """
    if scaler is None:
        return x_train, x_other
    num_features = x_train.shape[-1]
    scaler.fit(x_train.reshape(-1, num_features))
    scaled_train = scaler.transform(x_train.reshape(-1, num_features))
    scaled_other = scaler.transform(x_other.reshape(-1, num_features))
    return (
        scaled_train.reshape(x_train.shape),
        scaled_other.reshape(x_other.shape),
    )


def _build_model(
    config: dict[str, Any],
    model_cls: type[nn.Module],
    num_features: int,
    device: str,
    output_size: int = 1,
) -> nn.Module:
    """Construct and build a model from a hyperparameter config."""
    model = model_cls(
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        output_size=output_size,
    )
    return model.build(num_features).to(device)


def _target_width(y: np.ndarray) -> int:
    """Return the forecast horizon (target columns) of a target array."""
    array = np.asarray(y)
    return int(array.shape[1]) if array.ndim > 1 else 1


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
) -> float:
    """Run a single training epoch over `loader`, returning its mean loss."""
    model.train()
    total, batches = 0.0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        # Detach before converting: keeping the graph alive to read a scalar
        # warns and can retain memory.
        total += loss.detach().item()
        batches += 1
    return total / max(batches, 1)


def train_folds(
    config: dict[str, Any],
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    model_cls: type[nn.Module] = RNNForecaster,
    num_features: int | None = None,
    max_epochs: int = 50,
    device: str = "cpu",
    scaler_factory: ScalerFactory | None = None,
) -> Iterator[tuple[int, float, list[dict[str, torch.Tensor]]]]:
    """Train one model per fold in lockstep, yielding per-epoch results.

    Each yielded step advances every fold's model by one epoch and reports the
    mean validation RMSE across folds -- the quantity a scheduler prunes on.

    Args:
        config: Hyperparameters (``hidden_size``, ``num_layers``, ``dropout``,
            ``lr``, ``batch_size``, optional ``weight_decay``).
        x_dev: Development windows, shape ``(n, seq_len, num_features)``.
        y_dev: Development targets, shape ``(n, ...)``.
        splits: Fold indices from `walk_forward_splits`.
        model_cls: Model class to instantiate (baseline or augmented).
        num_features: Feature count; inferred from `x_dev` when ``None``.
        max_epochs: Number of epochs to train.
        device: Torch device string.
        scaler_factory: Optional per-fold scaler factory (fit on train only).

    Yields:
        ``(epoch, mean_val_rmse, fold_state_dicts)`` after each epoch.
    """
    num_features = num_features or x_dev.shape[-1]
    output_size = _target_width(y_dev)
    loss_fn = nn.MSELoss()
    folds = []
    for train_idx, val_idx in splits:
        x_train, x_val = _scale_features(
            scaler_factory() if scaler_factory else None,
            x_dev[train_idx],
            x_dev[val_idx],
        )
        loader = DataLoader(
            TensorDataset(_to_tensor(x_train), _to_tensor(y_dev[train_idx])),
            batch_size=config["batch_size"],
            shuffle=True,
        )
        folds.append(
            {
                "model": _build_model(
                    config, model_cls, num_features, device, output_size
                ),
                "loader": loader,
                "x_val": _to_tensor(x_val).to(device),
                "y_val": _to_tensor(y_dev[val_idx]).to(device),
            }
        )
    for fold in folds:
        fold["optimizer"] = torch.optim.Adam(
            fold["model"].parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.0),
        )

    for epoch in range(max_epochs):
        val_rmses, states = [], []
        for fold in folds:
            _train_one_epoch(
                fold["model"], fold["loader"], fold["optimizer"], loss_fn, device
            )
            fold["model"].eval()
            with torch.no_grad():
                preds = fold["model"](fold["x_val"])
            val_rmses.append(rmse(preds, fold["y_val"]))
            states.append(fold["model"].state_dict())
        yield epoch, float(np.mean(val_rmses)), states


def default_param_space() -> dict[str, Any]:
    """Return the shared Ray Tune search space (used for both models)."""
    from ray import tune

    return {
        "hidden_size": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([1, 2, 3]),
        "dropout": tune.uniform(0.0, 0.5),
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([32, 64, 128]),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
    }


def _tune_trainable(
    config: dict[str, Any],
    *,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    model_cls: type[nn.Module],
    num_features: int,
    max_epochs: int,
    device: str,
    scaler_factory: ScalerFactory | None,
    checkpoint_freq: int = 0,
) -> None:
    """Ray Tune trainable: train folds and report mean val RMSE each epoch.

    Checkpoints are written every `checkpoint_freq` epochs, or never when it is
    0 (the default). ASHA only needs the reported metric, and the best
    configuration is retrained from scratch by `evaluate_on_test`, so trial
    checkpoints are optional -- see `tune_model` for the trade-off.
    """
    from ray import tune

    for epoch, mean_rmse, states in train_folds(
        config,
        x_dev,
        y_dev,
        splits,
        model_cls=model_cls,
        num_features=num_features,
        max_epochs=max_epochs,
        device=device,
        scaler_factory=scaler_factory,
    ):
        metrics = {"val_rmse": mean_rmse, "epoch": epoch}
        if checkpoint_freq and (epoch + 1) % checkpoint_freq == 0:
            with tempfile.TemporaryDirectory() as ckpt_dir:
                torch.save(
                    {"epoch": epoch, "fold_states": states, "config": config},
                    os.path.join(ckpt_dir, "checkpoint.pt"),
                )
                tune.report(
                    metrics, checkpoint=tune.Checkpoint.from_directory(ckpt_dir)
                )
        else:
            tune.report(metrics)


def tune_model(
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    *,
    model_cls: type[nn.Module] = RNNForecaster,
    num_features: int | None = None,
    n_folds: int = 4,
    val_size: int | None = None,
    max_epochs: int = 50,
    num_samples: int = 20,
    grace_period: int = 5,
    param_space: dict[str, Any] | None = None,
    storage_path: str = DRIVE_MODELS_PATH,
    experiment_name: str = "rnn_tune",
    gpu_per_trial: float = 0.0,
    device: str | None = None,
    scaler_factory: ScalerFactory | None = None,
    checkpoint_freq: int = 0,
    seed: int = 0,
) -> Any:
    """Tune a model with Optuna (TPE) search + ASHA over walk-forward folds.

    The same call tunes the baseline and augmented models -- pass the matching
    `model_cls` and features; everything else stays identical for a fair
    comparison. Results and checkpoints persist under `storage_path` (the Shared
    Drive by default) so a Colab disconnect can be resumed.

    Args:
        x_dev: Development windows, shape ``(n, seq_len, num_features)``.
        y_dev: Development targets.
        model_cls: Model class to tune.
        num_features: Feature count; inferred from `x_dev` when ``None``.
        n_folds: Number of expanding walk-forward folds.
        val_size: Samples per validation block (see `walk_forward_splits`).
        max_epochs: Max epochs per trial (ASHA ``max_t``).
        num_samples: Number of hyperparameter configurations to try.
        grace_period: Minimum epochs before ASHA may stop a trial.
        param_space: Search space; defaults to `default_param_space`.
        storage_path: Directory for Tune results/checkpoints.
        experiment_name: Run name (used for resuming).
        gpu_per_trial: Fractional GPU per trial (e.g. 0.25 to pack four).
        device: Torch device; auto-detected when ``None``.
        scaler_factory: Optional per-fold scaler factory.
        checkpoint_freq: Write a trial checkpoint every this many epochs; 0
            (default) disables them. ASHA prunes on the reported metric alone
            and `evaluate_on_test` retrains the winning config from scratch, so
            checkpoints are not needed for the normal flow -- and writing them
            every epoch to a Drive-backed `storage_path` forces a full
            experiment-state snapshot each time, which Ray warns about and which
            dominates runtime for short trials. Set to 1 for mid-trial
            resumability or for a future Population Based Training scheduler.
        seed: Seed for the search algorithm and RNGs.

    Returns:
        The Ray Tune ``ResultGrid`` for the run. Call
        ``result.get_best_result("val_rmse", mode="min")`` for the best trial,
        ``result.get_dataframe()`` for a per-trial summary, and each result's
        ``metrics_dataframe`` for per-epoch learning curves.
    """
    # Import the config classes from ray.tune, not ray.train: under Ray Train V2
    # (default since 2.43) ray.train.RunConfig is a different class whose
    # deprecated `verbose` field is a string sentinel, which Tuner then feeds to
    # get_air_verbosity() and crashes on.
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    set_seed(seed)
    num_features = num_features or x_dev.shape[-1]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = walk_forward_splits(len(x_dev), n_folds=n_folds, val_size=val_size)

    trainable = tune.with_parameters(
        _tune_trainable,
        x_dev=x_dev,
        y_dev=y_dev,
        splits=splits,
        model_cls=model_cls,
        num_features=num_features,
        max_epochs=max_epochs,
        device=device,
        scaler_factory=scaler_factory,
        checkpoint_freq=checkpoint_freq,
    )
    trainable = tune.with_resources(trainable, {"cpu": 1, "gpu": gpu_per_trial})

    # num_to_keep forces an experiment-state snapshot each time a trial exceeds
    # it, so only constrain retention when checkpoints are actually written.
    checkpoint_config = tune.CheckpointConfig(
        num_to_keep=2 if checkpoint_freq else None,
        checkpoint_score_attribute="val_rmse",
        checkpoint_score_order="min",
    )

    tuner = tune.Tuner(
        trainable,
        param_space=param_space or default_param_space(),
        tune_config=tune.TuneConfig(
            scheduler=ASHAScheduler(
                time_attr="training_iteration",
                max_t=max_epochs,
                grace_period=grace_period,
            ),
            search_alg=OptunaSearch(seed=seed),
            metric="val_rmse",
            mode="min",
            num_samples=num_samples,
        ),
        run_config=tune.RunConfig(
            name=experiment_name,
            storage_path=storage_path,
            checkpoint_config=checkpoint_config,
        ),
    )
    return tuner.fit()


def evaluate_on_test(
    config: dict[str, Any],
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    model_cls: type[nn.Module] = RNNForecaster,
    num_features: int | None = None,
    max_epochs: int = 50,
    device: str | None = None,
    scaler_factory: ScalerFactory | None = None,
) -> dict[str, Any]:
    """Retrain `config` on all development data and score the held-out test set.

    Args:
        config: The chosen hyperparameter configuration.
        x_dev: All development windows (train + val periods).
        y_dev: Development targets.
        x_test: Held-out test windows (chronologically after `x_dev`).
        y_test: Test targets.
        model_cls: Model class to train.
        num_features: Feature count; inferred from `x_dev` when ``None``.
        max_epochs: Epochs to train the final model.
        device: Torch device; auto-detected when ``None``.
        scaler_factory: Optional scaler (fit on development data only).

    Returns:
        A dict with ``test_rmse`` and ``test_directional_accuracy`` floats, the
        raw ``predictions`` / ``targets`` arrays, the per-epoch
        ``train_loss_history``, and the fitted ``model`` and ``scaler`` (the
        scaler is ``None`` when no `scaler_factory` was given) so the run can be
        persisted.
    """
    num_features = num_features or x_dev.shape[-1]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    scaler = scaler_factory() if scaler_factory else None
    x_dev_s, x_test_s = _scale_features(scaler, x_dev, x_test)
    loader = DataLoader(
        TensorDataset(_to_tensor(x_dev_s), _to_tensor(y_dev)),
        batch_size=config["batch_size"],
        shuffle=True,
    )
    model = _build_model(config, model_cls, num_features, device, _target_width(y_dev))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    loss_fn = nn.MSELoss()
    train_loss_history = [
        _train_one_epoch(model, loader, optimizer, loss_fn, device)
        for _ in range(max_epochs)
    ]

    model.eval()
    with torch.no_grad():
        preds = model(_to_tensor(x_test_s).to(device)).cpu()
    targets = _to_tensor(y_test)
    # Naive baselines travel with every result: on daily returns a model can
    # post a small RMSE while being no better than predicting zero.
    summary = summarize_forecast(targets.numpy(), preds.numpy())
    return {
        "test_rmse": summary["rmse"],
        "test_directional_accuracy": summary["directional_accuracy"],
        "naive_zero_rmse": summary["naive_zero_rmse"],
        "naive_previous_rmse": summary["naive_previous_rmse"],
        "skill_vs_zero": summary["skill_vs_zero"],
        "skill_vs_previous": summary["skill_vs_previous"],
        "beats_naive": summary["beats_naive"],
        "predictions": preds.numpy(),
        "targets": targets.numpy(),
        "train_loss_history": train_loss_history,
        "model": model,
        "scaler": scaler,
    }
