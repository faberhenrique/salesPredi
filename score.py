"""
Azure ML scoring script for the sales value predictor.

Usage inside Azure ML:
    - The workspace should mount or copy `sales_value_model.keras` and `model_meta.json`
      to the working directory (or set env vars MODEL_PATH / MODEL_META_PATH).
    - Azure ML will import this file and call `init()` once, then `run(data)` for each request.

Expected request payload (JSON):
{
  "data": [
    {"date": "2024-05-01", "value": 1234.5, "num_items": 87},
    {"date": "2024-05-02", "value": 1560.0, "num_items": 92},
    ...
  ]
}

The list must contain at least (lags + 1) sequential daily records so that
lag features can be built for the most recent day. The scorer will return
the predicted sales value for the day AFTER the latest date provided.
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

MODEL_PATH = os.getenv("MODEL_PATH", "sales_value_model.keras")
META_PATH = os.getenv("MODEL_META_PATH", "model_meta.json")

_model: Optional[tf.keras.Model] = None
_feature_names: Optional[List[str]] = None
_lags: Optional[int] = None


def _load_artifacts() -> None:
    """Load model and metadata once at startup."""
    global _model, _feature_names, _lags

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'")
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _feature_names is None or _lags is None:
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Metadata file not found at '{META_PATH}'")
        with open(META_PATH, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        _feature_names = meta.get("feature_names")
        if not _feature_names:
            raise ValueError("Metadata missing 'feature_names'")
        _lags = int(meta.get("lags", len(_feature_names) // 2))


def _build_feature_matrix(records: List[Dict[str, Any]]) -> np.ndarray:
    """Convert raw daily records into the feature vector expected by the model."""
    if _feature_names is None or _lags is None:
        raise RuntimeError("Artifacts not loaded")

    df = pd.DataFrame(records)
    required_cols = {"date", "value", "num_items"}
    if not required_cols.issubset(df.columns):
        missing = required_cols.difference(df.columns)
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if len(df) < _lags + 1:
        raise ValueError(f"Need at least {_lags + 1} rows to compute {_lags} lags")

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7.0)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7.0)
    df["mon_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["mon_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    df["t_idx"] = (df.index - df.index.min()) / (df.index.max() - df.index.min() + 1e-9)

    for k in range(1, _lags + 1):
        df[f"value_lag_{k}"] = df["value"].shift(k)
        df[f"items_lag_{k}"] = df["num_items"].shift(k)

    latest = df.iloc[-1]
    feature_vector = [latest[name] for name in _feature_names]

    if any(pd.isna(feature_vector)):
        raise ValueError(
            "Found NaN in generated features; check if enough historical rows were provided."
        )

    return np.asarray(feature_vector, dtype="float32").reshape(1, -1)


def init() -> None:
    """Azure ML entry point: load artifacts before handling requests."""
    _load_artifacts()


def run(raw_data: Any) -> Dict[str, Any]:
    """Azure ML entry point: make a prediction from raw input."""
    try:
        _load_artifacts()

        if isinstance(raw_data, str):
            payload = json.loads(raw_data)
        else:
            payload = raw_data

        if not isinstance(payload, dict):
            raise ValueError("Payload must be a JSON object with a 'data' field")

        records = payload.get("data")
        if records is None:
            raise ValueError("Payload missing 'data' field")
        if isinstance(records, dict):
            # Allow both {"data": {...}} and {"data": [{...}]}
            records = [records]
        if not isinstance(records, list):
            raise ValueError("'data' must be a list of daily records")

        features = _build_feature_matrix(records)
        if _model is None:
            raise RuntimeError("Model failed to load")
        prediction = float(_model.predict(features, verbose=0).ravel()[0])

        latest_dates = pd.to_datetime([r.get("date") for r in records])
        max_ts = latest_dates.max() if len(latest_dates) else None
        last_known = (
            max_ts.date().isoformat() if max_ts is not None and not pd.isna(max_ts) else None
        )

        return {
            "prediction": prediction,
            "last_known_date": last_known,
            "message": "Prediction corresponds to the day after 'last_known_date'.",
        }
    except Exception as exc:  # pylint: disable=broad-except
        return {"error": str(exc)}


# Allow local testing via `python score.py --json payload.json`
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Local scorer test harness.")
    parser.add_argument("--json", type=str, help="Path to JSON payload file.")
    args = parser.parse_args()

    init()

    if args.json:
        with open(args.json, "r", encoding="utf-8") as fh:
            content = fh.read()
    else:
        content = input("Paste JSON payload:\n")

    print(run(content))
