"""Persist and reload a complete model run on the Shared Drive.

A tuning run on Colab is expensive and the session is ephemeral, so everything
needed to reuse, audit, or compare a run is written to one self-describing
directory under `DRIVE_MODELS_PATH`:

    <model_name>_<timestamp>/
        model.pt              state_dict plus the args needed to rebuild it
        scaler.joblib         fitted feature scaler (only if one was used)
        tuning_trials.csv     one row per hyperparameter trial
        learning_curves.csv   long form: trial, epoch, val_rmse
        test_predictions.csv  date, target_*, pred_*
        manifest.json         metrics, config, and run provenance

A `<model_name>_latest.json` pointer at the models root records the newest run,
so the evaluation notebook can find it without hardcoding a timestamp.

Weights are stored as a ``state_dict`` with explicit build arguments rather than
a pickled module, so they survive refactors. `RNNForecaster` builds lazily, so
`load_run` rebuilds the architecture before loading the weights.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from config import DRIVE_MODELS_PATH
from src.models.rnn_forecaster import RNNForecaster

MODEL_FILENAME = "model.pt"
SCALER_FILENAME = "scaler.joblib"
TRIALS_FILENAME = "tuning_trials.csv"
CURVES_FILENAME = "learning_curves.csv"
PREDICTIONS_FILENAME = "test_predictions.csv"
MANIFEST_FILENAME = "manifest.json"


def _git_commit() -> str | None:
    """Return the current git commit hash, or ``None`` outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _build_args(model: nn.Module) -> dict[str, Any]:
    """Extract the arguments needed to reconstruct `model`.

    Args:
        model: A built model exposing the `RNNForecaster` hyperparameters.

    Returns:
        A dict of constructor arguments plus ``num_features``.

    Raises:
        ValueError: If the model has not been built yet (no feature count).
    """
    num_features = getattr(model, "num_features", None)
    if num_features is None:
        raise ValueError("Model is not built; cannot record its architecture.")
    return {
        "num_features": int(num_features),
        "hidden_size": int(model.hidden_size),
        "num_layers": int(model.num_layers),
        "dropout": float(model.dropout),
        "output_size": int(model.output_size),
    }


def _predictions_frame(
    predictions: np.ndarray,
    targets: np.ndarray,
    dates: Sequence[Any] | None,
) -> pd.DataFrame:
    """Build a tidy frame of test predictions and targets."""
    preds = np.asarray(predictions).reshape(len(predictions), -1)
    actuals = np.asarray(targets).reshape(len(targets), -1)
    frame = pd.DataFrame(
        {
            **{f"target_{k}": actuals[:, k] for k in range(actuals.shape[1])},
            **{f"pred_{k}": preds[:, k] for k in range(preds.shape[1])},
        }
    )
    if dates is not None:
        frame.insert(0, "date", pd.to_datetime(pd.Series(list(dates))))
    return frame


def _curves_frame(curves: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-trial learning curves into one long-form frame."""
    parts = []
    for i, curve in enumerate(curves):
        part = curve.copy()
        if "trial" not in part.columns:
            part.insert(0, "trial", i)
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def save_run(
    model_name: str,
    model: nn.Module,
    config: dict[str, Any],
    metrics: dict[str, Any],
    *,
    scaler: Any | None = None,
    trials: pd.DataFrame | None = None,
    learning_curves: Sequence[pd.DataFrame] | None = None,
    predictions: np.ndarray | None = None,
    targets: np.ndarray | None = None,
    dates: Sequence[Any] | None = None,
    train_loss_history: Sequence[float] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    models_dir: str | Path = DRIVE_MODELS_PATH,
) -> Path:
    """Write a complete run to a timestamped directory and update the pointer.

    Args:
        model_name: Short run label, e.g. ``"baseline"`` or ``"augmented"``.
        model: The trained (built) model whose weights should be saved.
        config: The winning hyperparameter configuration.
        metrics: Headline metrics, e.g. test RMSE and directional accuracy.
        scaler: Fitted feature scaler, if one was used during training.
        trials: Per-trial tuning summary (``ResultGrid.get_dataframe()``).
        learning_curves: Per-trial per-epoch metric frames.
        predictions: Test-set predictions.
        targets: Test-set actuals.
        dates: Dates aligned with `predictions` / `targets`.
        train_loss_history: Per-epoch training loss of the final model.
        extra_metadata: Additional provenance (dataset file, split sizes, seed).
        models_dir: Root directory for runs (defaults to the Shared Drive).

    Returns:
        The path of the run directory that was written.
    """
    root = Path(models_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    build_args = _build_args(model)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "build_args": build_args,
            "model_class": type(model).__name__,
            "config": config,
        },
        run_dir / MODEL_FILENAME,
    )

    if scaler is not None:
        import joblib

        joblib.dump(scaler, run_dir / SCALER_FILENAME)

    if trials is not None:
        trials.to_csv(run_dir / TRIALS_FILENAME, index=False)

    if learning_curves:
        _curves_frame(learning_curves).to_csv(run_dir / CURVES_FILENAME, index=False)

    if predictions is not None and targets is not None:
        _predictions_frame(predictions, targets, dates).to_csv(
            run_dir / PREDICTIONS_FILENAME, index=False
        )

    manifest = {
        "model_name": model_name,
        "timestamp": timestamp,
        "metrics": {
            key: (float(value) if isinstance(value, (int, float)) else value)
            for key, value in metrics.items()
        },
        "config": config,
        "build_args": build_args,
        "train_loss_history": list(train_loss_history or []),
        "has_scaler": scaler is not None,
        "provenance": {
            "git_commit": _git_commit(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "cuda_available": torch.cuda.is_available(),
            **(extra_metadata or {}),
        },
    }
    (run_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    pointer = {"run_dir": str(run_dir), "timestamp": timestamp}
    (root / f"{model_name}_latest.json").write_text(
        json.dumps(pointer, indent=2), encoding="utf-8"
    )
    return run_dir


def find_latest_run(
    model_name: str,
    models_dir: str | Path = DRIVE_MODELS_PATH,
) -> Path:
    """Return the directory of the most recently saved run for `model_name`.

    Args:
        model_name: The label passed to `save_run`.
        models_dir: Root directory for runs.

    Returns:
        The run directory path.

    Raises:
        FileNotFoundError: If no pointer or run directory exists.
    """
    root = Path(models_dir)
    pointer = root / f"{model_name}_latest.json"
    if pointer.exists():
        run_dir = Path(json.loads(pointer.read_text(encoding="utf-8"))["run_dir"])
        if run_dir.exists():
            return run_dir
    candidates = sorted(root.glob(f"{model_name}_*/"))
    if not candidates:
        raise FileNotFoundError(f"No saved runs for {model_name!r} under {root}.")
    return candidates[-1]


def load_run(
    run_dir: str | Path,
    *,
    model_cls: type[nn.Module] | None = RNNForecaster,
    device: str = "cpu",
) -> dict[str, Any]:
    """Reload a saved run, optionally rebuilding the model and its weights.

    Args:
        run_dir: Directory written by `save_run`.
        model_cls: Class to reconstruct (must accept the saved build args), or
            ``None`` to skip rebuilding and return ``model=None``. Reconstruction
            assumes `RNNForecaster`'s build args (``hidden_size``, ``num_layers``,
            ...) and a ``build`` method; the PatchTST stages record placeholder
            zeros for those, so their runs must be loaded with ``None``. Reading
            the rest of the run never needs the class -- `save_run` stores a plain
            dict of tensors, not a pickled model.
        device: Device to map the weights onto.

    Returns:
        A dict with the rebuilt ``model`` (``None`` when `model_cls` is ``None``),
        plus ``config``, ``manifest``, and whichever of ``scaler``, ``trials``,
        ``learning_curves`` and ``predictions`` were saved (``None`` when absent).
    """
    path = Path(run_dir)
    payload = torch.load(path / MODEL_FILENAME, map_location=device, weights_only=False)

    model = None
    if model_cls is not None:
        build_args = payload["build_args"]
        model = model_cls(
            hidden_size=build_args["hidden_size"],
            num_layers=build_args["num_layers"],
            dropout=build_args["dropout"],
            output_size=build_args["output_size"],
        )
        model.build(build_args["num_features"]).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()

    scaler = None
    if (path / SCALER_FILENAME).exists():
        import joblib

        scaler = joblib.load(path / SCALER_FILENAME)

    def _read(name: str) -> pd.DataFrame | None:
        target = path / name
        return pd.read_csv(target) if target.exists() else None

    manifest_path = path / MANIFEST_FILENAME
    return {
        "model": model,
        "config": payload.get("config", {}),
        "manifest": (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        ),
        "scaler": scaler,
        "trials": _read(TRIALS_FILENAME),
        "learning_curves": _read(CURVES_FILENAME),
        "predictions": _read(PREDICTIONS_FILENAME),
    }
