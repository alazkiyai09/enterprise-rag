"""Minimal citation registry for unified RAG responses."""

from dataclasses import dataclass, field


@dataclass
class CitationTracker:
    citations: list[dict[str, str]] = field(default_factory=list)

    def add(self, source: str, excerpt: str) -> None:
        self.citations.append({"source": source, "excerpt": excerpt})

    def export(self) -> list[dict[str, str]]:
        return list(self.citations)

