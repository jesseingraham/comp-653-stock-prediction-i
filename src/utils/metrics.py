"""Forecast metrics and the naive baselines every result must be judged against.

Daily log returns are close to unpredictable, so a model can post a small RMSE
while adding no value at all: predicting a constant zero return already scores
roughly the standard deviation of the targets. Reporting RMSE alone therefore
hides whether a model learned anything.

`summarize_forecast` is the entry point -- it reports the model's error next to
two naive references and the skill score against each, so a result cannot be
quoted without showing whether it beat doing nothing:

    * **zero baseline** -- always predict a 0 return (the "no information" case).
    * **previous-return baseline** -- predict the last observed return (a
      persistence forecast, which also catches a model that has merely learned
      to echo its input).

This matters doubly for the pattern experiment: the augmented model only
supports a claim about chart patterns if *both* arms clear these baselines
first. Otherwise a difference between them is noise on top of nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# A forecast whose spread is below this fraction of the targets' spread is
# treated as collapsed: it is emitting a near-constant value rather than
# discriminating between days.
COLLAPSE_STD_RATIO = 0.05


def _to_numpy(values: Any) -> np.ndarray:
    """Convert a torch tensor, sequence, or array to a float NumPy array."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def rmse(y_pred: Any, y_true: Any) -> float:
    """Root mean squared error between predictions and actuals."""
    pred, true = _to_numpy(y_pred), _to_numpy(y_true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(y_pred: Any, y_true: Any) -> float:
    """Mean absolute error between predictions and actuals."""
    pred, true = _to_numpy(y_pred), _to_numpy(y_true)
    return float(np.mean(np.abs(pred - true)))


def directional_accuracy(y_pred: Any, y_true: Any) -> float:
    """Fraction of predictions with the correct up/down sign.

    Targets are signed returns, so direction is simply the sign. Flattens across
    samples and all forecast horizons.

    Args:
        y_pred: Predicted return targets, shape ``(n,)`` or ``(n, horizon)``.
        y_true: Actual return targets, same shape as `y_pred`.

    Returns:
        Directional accuracy in ``[0, 1]`` (0.0 if empty).
    """
    pred, true = _to_numpy(y_pred).ravel(), _to_numpy(y_true).ravel()
    if true.size == 0:
        return 0.0
    return float(np.mean(np.sign(pred) == np.sign(true)))


def naive_zero_forecast(y_true: Any) -> np.ndarray:
    """Return the all-zero forecast: the "predict no change" reference."""
    return np.zeros_like(_to_numpy(y_true))


def naive_previous_forecast(y_true: Any) -> np.ndarray:
    """Return the persistence forecast: each step predicts the previous return.

    The first sample has no predecessor and falls back to zero.

    Args:
        y_true: Actual returns, shape ``(n,)`` or ``(n, horizon)``, in
            chronological order.

    Returns:
        An array shaped like `y_true` holding the previous sample's values.
    """
    true = _to_numpy(y_true)
    shifted = np.zeros_like(true)
    if len(true) > 1:
        shifted[1:] = true[:-1]
    return shifted


def skill_score(model_error: float, baseline_error: float) -> float:
    """Fractional error reduction versus a baseline.

    Args:
        model_error: The model's error (e.g. RMSE).
        baseline_error: The baseline's error, in the same units.

    Returns:
        ``1 - model/baseline``: positive means the model beats the baseline,
        0.0 means no better, negative means worse. Returns 0.0 when the
        baseline error is 0.
    """
    if baseline_error == 0:
        return 0.0
    return float(1.0 - model_error / baseline_error)


def summarize_forecast(y_true: Any, y_pred: Any) -> dict[str, float | bool]:
    """Score a forecast against both naive baselines.

    Args:
        y_true: Actual return targets, chronologically ordered.
        y_pred: Model predictions, same shape as `y_true`.

    Returns:
        A dict with the model's ``rmse``, ``mae`` and ``directional_accuracy``;
        the ``naive_zero_rmse`` and ``naive_previous_rmse`` references; the
        ``skill_vs_zero`` / ``skill_vs_previous`` scores; and ``beats_naive``,
        which is True only when the model beats *both* baselines.
    """
    model_rmse = rmse(y_pred, y_true)
    zero_rmse = rmse(naive_zero_forecast(y_true), y_true)
    previous_rmse = rmse(naive_previous_forecast(y_true), y_true)
    pred_std = float(_to_numpy(y_pred).std())
    true_std = float(_to_numpy(y_true).std())
    return {
        "rmse": model_rmse,
        "mae": mae(y_pred, y_true),
        "directional_accuracy": directional_accuracy(y_pred, y_true),
        "naive_zero_rmse": zero_rmse,
        "naive_previous_rmse": previous_rmse,
        "skill_vs_zero": skill_score(model_rmse, zero_rmse),
        "skill_vs_previous": skill_score(model_rmse, previous_rmse),
        "beats_naive": bool(model_rmse < zero_rmse and model_rmse < previous_rmse),
        "prediction_std": pred_std,
        "target_std": true_std,
        "prediction_std_ratio": (pred_std / true_std) if true_std else 0.0,
        "collapsed": bool(true_std > 0 and pred_std < COLLAPSE_STD_RATIO * true_std),
    }


def format_summary(summary: dict[str, float | bool]) -> str:
    """Render `summarize_forecast` output as an aligned, readable block."""
    verdict = (
        "BEATS both naive baselines"
        if summary["beats_naive"]
        else "does NOT beat the naive baselines"
    )
    lines = [
        f"RMSE                 {summary['rmse']:.6f}",
        f"MAE                  {summary['mae']:.6f}",
        f"Directional accuracy {summary['directional_accuracy']:.3f}",
        f"Naive zero RMSE      {summary['naive_zero_rmse']:.6f}  "
        f"(skill {summary['skill_vs_zero']:+.3f})",
        f"Naive previous RMSE  {summary['naive_previous_rmse']:.6f}  "
        f"(skill {summary['skill_vs_previous']:+.3f})",
        f"Prediction spread    {summary['prediction_std']:.6f} vs target "
        f"{summary['target_std']:.6f} "
        f"(ratio {summary['prediction_std_ratio']:.3f})",
        f"Verdict: model {verdict}.",
    ]
    if summary["collapsed"]:
        lines.append(
            "WARNING: predictions are near-constant, so the model is not "
            "discriminating between days. Directional accuracy is then an "
            "artifact of the up-day rate, not skill."
        )
    return "\n".join(lines)
