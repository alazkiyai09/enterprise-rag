"""SQL generation shell for the unified repo."""

from dataclasses import dataclass


@dataclass
class SQLGenerationResult:
    question: str
    sql: str
    notes: str


class SQLGenerator:
    def generate(self, question: str) -> SQLGenerationResult:
        return SQLGenerationResult(
            question=question,
            sql="-- connect a database-backed text-to-SQL engine to enable execution",
            notes="Migrated from DataChat-RAG as part of the unified enterprise-rag layout.",
        )

