"""Thin SQL mode facade over the migrated DataChat routing layer."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass
class SQLRoutingPreview:
    query_type: str
    confidence: float
    reasoning: str | None = None


class SQLRetriever:
    def classify(self, question: str) -> SQLRoutingPreview:
        router_cls = getattr(
            import_module("src.datachat.routers.query_router"),
            "QueryRouter",
        )
        classification = router_cls().classify(question)
        return SQLRoutingPreview(
            query_type=str(classification.query_type.value),
            confidence=float(classification.confidence),
            reasoning=getattr(classification, "reasoning", None),
        )

    def query(self, question: str, tables: list[str] | None = None) -> dict[str, Any]:
        return {
            "status": "not_configured",
            "question": question,
            "tables": tables or [],
            "message": "The SQL wrapper is present; connect database credentials before execution.",
        }

