from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class RatingResult:
    cf_scaled_difficulty: float
    confidence: float          # 0.0–1.0
    estimated: bool = True     # always True — research premise (Decision 2)
    model_name: str = ""


class BaseRatingPredictor(ABC):
    """Plugin interface for difficulty predictors.

    To add a new LLM/model predictor:
    1. Subclass BaseRatingPredictor
    2. Implement predict()
    3. Register it in config.yaml under rating_predictors
    """

    @abstractmethod
    def predict(self, problem_name: str, tags: list[str], platform: str,
                solved_count: int | None) -> RatingResult:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
