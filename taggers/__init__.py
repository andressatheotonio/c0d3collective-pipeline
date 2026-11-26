from .base import BaseTagger, TagResult
from .similarity_tagger import SimilarityTagger
from .ensemble import EnsembleTagger

__all__ = ["BaseTagger", "TagResult", "SimilarityTagger", "EnsembleTagger"]
