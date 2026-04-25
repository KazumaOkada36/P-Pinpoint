"""Location recommender — cosine similarity on normalized state feature vectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from pinpoint.recommender.features import build_query_vector, FEATURE_LABELS


@dataclass
class Recommendation:
    rank: int
    state: str
    state_name: str
    score: float
    # (feature_name, normalized_value, pct_contribution)
    top_features: list[tuple[str, float, float]] = field(default_factory=list)

    def feature_summary(self) -> str:
        """Human-readable summary of top contributing features."""
        parts = []
        for feat, val, pct in self.top_features:
            label = FEATURE_LABELS.get(feat, feat.replace("_", " ").title())
            parts.append(f"{label} ({pct:.0f}% weight)")
        return "; ".join(parts) if parts else "general economic strength"


class LocationRecommender:
    """Cosine-similarity recommender over state GDP features."""

    def __init__(self) -> None:
        self._scaler = MinMaxScaler()
        self._feature_matrix: Optional[np.ndarray] = None
        self._feature_columns: list[str] = []
        self._state_index: list[str] = []
        self._state_names: dict[str, str] = {}
        self._fitted = False

    def fit(self, feature_df: pd.DataFrame) -> "LocationRecommender":
        drop_cols = ["state_name"]
        numeric_df = feature_df.drop(columns=[c for c in drop_cols if c in feature_df.columns])
        numeric_df = numeric_df.select_dtypes(include=[np.number]).fillna(0.0)

        self._state_index = list(numeric_df.index)
        self._feature_columns = list(numeric_df.columns)
        self._state_names = {}
        if "state_name" in feature_df.columns:
            for state, name in zip(feature_df.index, feature_df["state_name"]):
                self._state_names[state] = name

        X = numeric_df.values.astype(float)
        self._feature_matrix = self._scaler.fit_transform(X)
        self._fitted = True
        return self

    def recommend(
        self,
        priorities: list[str],
        top_n: int = 5,
        preferred_region: Optional[str] = None,
    ) -> list[Recommendation]:
        if not self._fitted or self._feature_matrix is None:
            raise RuntimeError("Model not fitted. Run `pinpoint train` first.")

        query = build_query_vector(priorities, self._feature_columns)
        query_scaled = self._scaler.transform(query.reshape(1, -1))
        scores = cosine_similarity(query_scaled, self._feature_matrix)[0]

        ranked_idx = np.argsort(scores)[::-1]

        # Normalise contribution percentages across all features
        total_query_weight = query.sum() or 1.0

        results: list[Recommendation] = []
        rank = 1
        for idx in ranked_idx:
            if rank > top_n:
                break
            state = self._state_index[idx]
            score = float(scores[idx])

            state_vec = self._feature_matrix[idx]
            # Contribution = query_weight * state_value
            contributions = query * state_vec
            total_contribution = contributions.sum() or 1.0

            top_feat_idx = np.argsort(contributions)[::-1][:3]
            top_features = [
                (
                    self._feature_columns[i],
                    float(state_vec[i]),
                    float(contributions[i] / total_contribution * 100),
                )
                for i in top_feat_idx
                if contributions[i] > 0
            ]

            results.append(
                Recommendation(
                    rank=rank,
                    state=state,
                    state_name=self._state_names.get(state, state),
                    score=round(score, 4),
                    top_features=top_features,
                )
            )
            rank += 1

        return results

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def feature_columns(self) -> list[str]:
        return self._feature_columns

    @property
    def n_states(self) -> int:
        return len(self._state_index)