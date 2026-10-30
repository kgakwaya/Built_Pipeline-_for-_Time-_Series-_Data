"""Chronological splitting, baselines, pipelines, time-aware tuning and evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------- splitting
def chronological_split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.70, val_frac: float = 0.15):
    """Split by time order (no shuffling): earliest -> train, next -> validation, latest -> test."""
    if not X.index.is_monotonic_increasing:
        raise ValueError("Data must be sorted chronologically before splitting.")
    n = len(X)
    i_tr, i_val = int(n * train_frac), int(n * (train_frac + val_frac))
    parts = {"train": slice(0, i_tr), "val": slice(i_tr, i_val), "test": slice(i_val, n)}
    out = {k: (X.iloc[s], y.iloc[s]) for k, s in parts.items()}
    assert_chronological_separation(out["train"][0].index, out["val"][0].index, out["test"][0].index)
    return out


def assert_chronological_separation(train_idx, val_idx, test_idx) -> None:
    """Raise unless max(train) < min(val) and max(val) < min(test)."""
    if not (train_idx.max() < val_idx.min() and val_idx.max() < test_idx.min()):
        raise AssertionError("Partitions are not chronologically separated.")


def split_summary(splits: dict) -> pd.DataFrame:
    """Date range and size of each partition."""
    rows = [{"partition": k, "start": X.index.min(), "end": X.index.max(), "n_rows": len(X),
             "target_mean": y.mean(), "target_std": y.std()} for k, (X, y) in splits.items()]
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- metrics
def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE, R2 and MAPE (MAPE is NaN if any true value is <= 0.05, where it would be misleading)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if np.all(y_true > 0.05) else float("nan")
    return {"MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)), "MAPE_pct": mape}


# ----------------------------------------------------------------------------- baselines / pipelines
def baseline_predict(X: pd.DataFrame, column: str) -> np.ndarray:
    """Baseline forecast copied from a feature that is known at the forecast origin.

    ``gap_lag1`` -> persistence; ``gap_lag24`` -> seasonal naive (same hour yesterday);
    ``gap_lag168`` -> seasonal naive (same hour last week).
    """
    return X[column].to_numpy(float)


def build_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=alpha))])


def build_rf_pipeline(seed: int = 42, **params) -> Pipeline:
    defaults = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "n_jobs": 1}
    defaults.update(params)
    return Pipeline([("model", RandomForestRegressor(random_state=seed, **defaults))])


# ----------------------------------------------------------------------------- tuning
def tune_model(pipeline: Pipeline, param_space: dict, X_train, y_train, *, search: str = "grid", n_splits: int = 4,
               n_iter: int = 10, seed: int = 42, scoring: str = "neg_mean_absolute_error"):
    """Hyper-parameter search with TimeSeriesSplit on the TRAINING partition only.

    Returns (fitted_search, cv_results_dataframe, summary_dict). The best estimator is refit on the full training set.
    """
    cv = TimeSeriesSplit(n_splits=n_splits)
    if search == "grid":
        searcher = GridSearchCV(pipeline, param_space, cv=cv, scoring=scoring, refit=True, n_jobs=1)
    elif search == "random":
        searcher = RandomizedSearchCV(pipeline, param_space, n_iter=n_iter, cv=cv, scoring=scoring,
                                      refit=True, random_state=seed, n_jobs=1)
    else:
        raise ValueError("search must be 'grid' or 'random'")
    t0 = time.perf_counter()
    searcher.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    results = pd.DataFrame(searcher.cv_results_)
    keep = [c for c in results.columns if c.startswith("param_")] + ["mean_test_score", "std_test_score", "rank_test_score"]
    results = results[keep].sort_values("rank_test_score").reset_index(drop=True)
    summary = {"search": search, "cv": f"TimeSeriesSplit(n_splits={n_splits})", "scoring": scoring,
               "n_candidates": int(len(results)), "n_fits": int(len(results) * n_splits),
               "best_params": searcher.best_params_, "best_cv_score": float(searcher.best_score_),
               "tuning_seconds": round(elapsed, 2)}
    return searcher, results, summary


# ----------------------------------------------------------------------------- experiments
def run_experiment(exp_id: str, model_name: str, splits: dict, *, estimator=None, baseline_column: str | None = None,
                   features: list[str] | None = None, feature_config: str = "", hyperparameters: str = "",
                   fitted: bool = False, extra_seconds: float = 0.0):
    """Fit (if needed) on train, then score on validation and test. Returns (result_row, predictions_dict).

    Baselines use ``baseline_column``; models use ``estimator`` with ``features``. If ``fitted`` is True the estimator
    is assumed already trained on the training partition (e.g. the refit best estimator of a search).
    """
    t0 = time.perf_counter()
    preds, row = {}, {"experiment": exp_id, "model": model_name, "features": feature_config, "hyperparameters": hyperparameters}
    X_tr, y_tr = splits["train"]
    if baseline_column is None:
        if not fitted:
            estimator.fit(X_tr[features], y_tr)
        for part in ("train", "val", "test"):
            preds[part] = estimator.predict(splits[part][0][features])
    else:
        for part in ("train", "val", "test"):
            preds[part] = baseline_predict(splits[part][0], baseline_column)
    runtime = time.perf_counter() - t0 + extra_seconds
    for part, label in (("train", "Train"), ("val", "Val"), ("test", "Test")):
        m = regression_metrics(splits[part][1], preds[part])
        for k in ("MAE", "RMSE", "R2"):
            row[f"{label}_{k}"] = m[k]
        if part == "test":
            row["Test_MAPE_pct"] = m["MAPE_pct"]
    row["runtime_s"] = round(runtime, 2)
    return row, preds


# ----------------------------------------------------------------------------- persistence
def save_model_bundle(model, metadata: dict, model_path: Path, metadata_path: Path) -> None:
    """Persist the fitted pipeline (joblib) and a JSON metadata file."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
