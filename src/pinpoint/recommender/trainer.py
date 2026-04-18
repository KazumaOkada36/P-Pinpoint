"""Train, save, and load the LocationRecommender."""

from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import Optional

import joblib

from pinpoint.agent.config import MODEL_FILE, CUSTOM_DATA_DIR
from pinpoint.recommender.features import build_feature_matrix
from pinpoint.recommender.model import LocationRecommender


def _default_data_dir() -> Path:
    """Find bundled Data_Set/clean_data relative to this package."""
    # Try relative to package root (installed or development)
    here = Path(__file__).resolve()
    # Traverse up to find Data_Set
    for parent in here.parents:
        candidate = parent / "Data_Set" / "clean_data"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find Data_Set/clean_data. "
        "Run from the P-Pinpoint repo root or set data_dir in config."
    )


def train(
    data_dir: Optional[Path] = None,
    custom_data_dir: Optional[Path] = None,
    quiet: bool = False,
) -> LocationRecommender:
    """
    Build feature matrix from CSVs, fit the recommender, save to disk.
    Returns the fitted model.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    if custom_data_dir is None:
        custom_data_dir = CUSTOM_DATA_DIR

    if not quiet:
        print(f"Loading data from: {data_dir}")

    feature_df = build_feature_matrix(data_dir, custom_data_dir)

    if not quiet:
        print(f"  {len(feature_df)} states | {len(feature_df.columns)} features")

    model = LocationRecommender()
    model.fit(feature_df)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    if not quiet:
        print(f"Model saved to: {MODEL_FILE}")

    return model


def load() -> Optional[LocationRecommender]:
    """Load previously trained model from disk. Returns None if not found."""
    if MODEL_FILE.exists():
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            return None
    return None


def load_or_train(
    data_dir: Optional[Path] = None,
    quiet: bool = True,
) -> LocationRecommender:
    """Load existing model or train a new one if not found."""
    model = load()
    if model is not None and model.is_fitted:
        return model
    return train(data_dir=data_dir, quiet=quiet)
