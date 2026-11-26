"""
Gradient boosting baseline predictor (free, no API key).
Features: solved_count, platform, number of tags, tag presence booleans.
Trained on the existing C0D3Collective dataset (problems with known cf_scaled_difficulty).
"""

import logging
import pickle
from pathlib import Path

import numpy as np

from .base import BaseRatingPredictor, RatingResult

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "ml_model.pkl"

KNOWN_TAGS = [
    "implementation", "math", "greedy", "dp", "data structures",
    "brute force", "constructive algorithms", "graphs", "sorting",
    "binary search", "strings", "trees", "number theory", "geometry",
    "bitmasks", "two pointers", "dfs and similar", "shortest paths",
    "probabilities", "divide and conquer",
]

PLATFORMS = ["codeforces", "atcoder", "kattis"]


def _featurize(problem_name: str, tags: list[str], platform: str,
               solved_count: int | None) -> np.ndarray:
    tag_set = set(tags)
    tag_flags = [1.0 if t in tag_set else 0.0 for t in KNOWN_TAGS]
    platform_flags = [1.0 if platform == p else 0.0 for p in PLATFORMS]
    sc = float(solved_count) if solved_count else 0.0
    return np.array([sc, len(tags)] + tag_flags + platform_flags, dtype=float)


class MLRatingPredictor(BaseRatingPredictor):
    def __init__(self):
        self._model = None

    @property
    def model_name(self) -> str:
        return "gradient_boosting_baseline"

    def _load_or_train(self, dataset_path: Path | None = None) -> None:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
            return

        if dataset_path is None:
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. "
                "Run pipeline.py with --train to build it from the dataset."
            )
        self._train(dataset_path)

    def _train(self, dataset_path: Path) -> None:
        import pandas as pd
        from sklearn.ensemble import GradientBoostingRegressor

        logger.info(f"Training gradient boosting model from {dataset_path}...")
        df = pd.read_csv(dataset_path)
        df = df[df["cf_scaled_difficulty"].notna() & (df["estimated"] != True)]

        X = np.array([
            _featurize(
                row.get("name", ""),
                str(row.get("tags", "")).split("|"),
                str(row.get("platform", "")),
                row.get("solved_count"),
            )
            for _, row in df.iterrows()
        ])
        y = df["cf_scaled_difficulty"].values

        model = GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42)
        model.fit(X, y)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        self._model = model
        logger.info("Model trained and saved.")

    def predict(self, problem_name: str, tags: list[str], platform: str,
                solved_count: int | None) -> RatingResult:
        if self._model is None:
            self._load_or_train()

        features = _featurize(problem_name, tags, platform, solved_count).reshape(1, -1)
        predicted = float(self._model.predict(features)[0])
        predicted = max(800.0, min(3500.0, predicted))

        return RatingResult(
            cf_scaled_difficulty=predicted,
            confidence=0.65,  # rough estimate; replaced when ensemble voting
            estimated=True,
            model_name=self.model_name,
        )
