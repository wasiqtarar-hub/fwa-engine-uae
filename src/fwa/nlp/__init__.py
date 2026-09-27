"""Document / NLP layer (Type T) — §4.9. Runs against a clearly-labelled
SYNTHETIC corpus, because the claim extract contains no documents."""

from dataclasses import dataclass
from typing import Any

from .synthetic import SyntheticCorpus, SyntheticDocument, SYNTHETIC_MARKER
from .pipeline import DocumentPipeline, DocumentAnalysis, Extraction, documentation_level_conflicts


@dataclass
class DocumentLayer:
    """What ``ControlContext.documents`` holds: the corpus plus its analyses."""

    corpus: SyntheticCorpus
    pipeline: DocumentPipeline

    @property
    def documents(self):
        return self.corpus.documents

    def summary(self) -> dict[str, Any]:
        return {"corpus": self.corpus.summary(), "pipeline": self.pipeline.summary()}


__all__ = [
    "SyntheticCorpus", "SyntheticDocument", "SYNTHETIC_MARKER",
    "DocumentPipeline", "DocumentAnalysis", "Extraction",
    "documentation_level_conflicts", "DocumentLayer",
]
