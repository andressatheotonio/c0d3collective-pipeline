"""
Semantic similarity tagger using sentence-transformers (free, local, no API).
Finds the k most similar tagged problems in the reference corpus and
assigns their tags weighted by cosine similarity.
"""

import logging
from collections import defaultdict

import numpy as np

from .base import BaseTagger, TagResult

logger = logging.getLogger(__name__)

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
MIN_SIMILARITY = 0.70  # only tags from problems with similarity ≥ this threshold


class SimilarityTagger(BaseTagger):
    def __init__(self):
        self._model = None
        self._corpus_embeddings: np.ndarray | None = None
        self._corpus_problems: list[dict] | None = None  # [{name, tags}, ...]

    @property
    def tagger_name(self) -> str:
        return "similarity_minilm"

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(EMBED_MODEL)
        logger.info(f"Loaded sentence-transformer model: {EMBED_MODEL}")

    def build_corpus(self, tagged_problems: list[dict]) -> None:
        """Pre-compute embeddings for all tagged problems in the reference corpus."""
        if self._model is None:
            self._load_model()

        self._corpus_problems = tagged_problems
        texts = [p["name"] for p in tagged_problems]
        self._corpus_embeddings = self._model.encode(texts, normalize_embeddings=True,
                                                      show_progress_bar=True)
        logger.info(f"Built corpus of {len(tagged_problems)} tagged problems.")

    def tag(self, problem_id: str, problem_name: str, platform: str,
            existing_tags: list[str]) -> TagResult:
        if existing_tags:
            return TagResult(
                tags=existing_tags,
                confidence=1.0,
                votes={t: 1.0 for t in existing_tags},
                estimated=False,
                tags_source="api",
            )

        if self._model is None:
            self._load_model()
        if self._corpus_embeddings is None or self._corpus_problems is None or len(self._corpus_problems) == 0:
            logger.warning(f"Corpus vazio — sem tags para {problem_id}")
            return TagResult(tags=[], confidence=0.0, votes={}, estimated=True, tags_source="similarity")

        query_embedding = self._model.encode([problem_name], normalize_embeddings=True)[0]
        similarities = self._corpus_embeddings @ query_embedding

        top_indices = np.argsort(similarities)[::-1][:TOP_K]
        tag_scores: dict[str, float] = defaultdict(float)
        total_weight = 0.0

        for idx in top_indices:
            sim = float(similarities[idx])
            if sim < MIN_SIMILARITY:
                break
            for tag in self._corpus_problems[idx]["tags"]:
                tag_scores[tag] += sim
            total_weight += sim

        if not tag_scores:
            return TagResult(tags=[], confidence=0.0, votes={}, estimated=True,
                             tags_source="similarity")

        max_score = max(tag_scores.values())
        votes = {t: s / total_weight for t, s in tag_scores.items()}
        selected = [t for t, s in sorted(tag_scores.items(), key=lambda x: -x[1])
                    if s / max_score >= 0.5]
        confidence = float(np.mean([votes[t] for t in selected])) if selected else 0.0

        return TagResult(
            tags=selected,
            confidence=confidence,
            votes=votes,
            estimated=True,
            tags_source="similarity",
        )
