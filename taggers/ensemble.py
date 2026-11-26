"""
Ensemble tagger: runs N independent taggers and combines their votes.
Voting strategy: confidence-weighted tag scores, majority threshold configurable.

To add a new tagger model, subclass BaseTagger, implement tag(), and register
it in config.yaml — no changes needed here.
"""

import logging
from collections import defaultdict

from .base import BaseTagger, TagResult

logger = logging.getLogger(__name__)


class EnsembleTagger:
    def __init__(self, taggers: list[BaseTagger], majority_threshold: float = 0.5):
        if not taggers:
            raise ValueError("EnsembleTagger requires at least one tagger.")
        self.taggers = taggers
        self.majority_threshold = majority_threshold

    def tag(self, problem_id: str, problem_name: str, platform: str,
            existing_tags: list[str]) -> TagResult:
        results = [
            t.tag(problem_id, problem_name, platform, existing_tags)
            for t in self.taggers
        ]

        # Short-circuit: if any tagger reports source="api", trust it directly
        for r in results:
            if r.tags_source == "api":
                return r

        # Aggregate votes, weighted by each tagger's confidence
        tag_scores: dict[str, float] = defaultdict(float)
        total_weight = sum(r.confidence for r in results) or 1.0

        for r in results:
            weight = r.confidence / total_weight
            for tag, vote_frac in r.votes.items():
                tag_scores[tag] += vote_frac * weight

        # Select tags that cross the majority threshold
        selected = [t for t, s in tag_scores.items() if s >= self.majority_threshold]
        ensemble_confidence = float(
            sum(tag_scores[t] for t in selected) / len(selected)
        ) if selected else 0.0

        votes = dict(tag_scores)
        source = results[0].tags_source if len(self.taggers) == 1 else "ensemble"

        logger.debug(
            f"[{problem_id}] ensemble → {len(selected)} tags "
            f"(confidence={ensemble_confidence:.2f}, source={source})"
        )

        return TagResult(
            tags=selected,
            confidence=ensemble_confidence,
            votes=votes,
            estimated=True,
            tags_source=source,
        )
