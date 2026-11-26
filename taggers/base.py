from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TagResult:
    tags: list[str]
    confidence: float              # 0.0–1.0 average across tags
    votes: dict[str, float]        # tag → fraction of models that voted for it
    estimated: bool = True         # always True — research premise (Decision 2)
    tags_source: str = ""          # "api" | "co_solving" | "similarity" | "llm"


class BaseTagger(ABC):
    """Plugin interface for topic taggers.

    To add a new LLM tagger:
    1. Subclass BaseTagger
    2. Implement tag()
    3. Register it in config.yaml under taggers
    """

    @abstractmethod
    def tag(self, problem_id: str, problem_name: str, platform: str,
            existing_tags: list[str]) -> TagResult:
        ...

    @property
    @abstractmethod
    def tagger_name(self) -> str:
        ...
