from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Problem:
    id: str                              # "{platform}_{problem_id}"
    platform: str                        # "codeforces" | "atcoder" | "kattis"
    problem_id: str                      # native platform id
    name: str
    url: str
    tags: list[str] = field(default_factory=list)
    rating: float | None = None          # native platform rating
    cf_scaled_difficulty: float | None = None  # normalized to CF scale (800–3500)
    solved_count: int | None = None
    contest_id: str | None = None
    fetched_at: datetime = field(default_factory=datetime.utcnow)
    estimated: bool = False              # True if any field was predicted


class BaseCollector(ABC):
    @abstractmethod
    def collect(self) -> list[Problem]:
        """Fetch all available problems from this platform."""
        ...

    @property
    @abstractmethod
    def platform_name(self) -> str:
        ...
