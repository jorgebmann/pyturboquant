"""pyturboquant.search -- Nearest-neighbor search with TurboQuant."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyturboquant.search.index import TurboQuantIndex

if TYPE_CHECKING:
    from pyturboquant.search.langchain import TurboQuantVectorStore

__all__ = ["TurboQuantIndex", "TurboQuantVectorStore"]


def __getattr__(name: str) -> Any:
    # Lazy import so importing pyturboquant.search doesn't pull in langchain-core
    # unless the optional integration is actually requested.
    if name == "TurboQuantVectorStore":
        from pyturboquant.search.langchain import TurboQuantVectorStore

        return TurboQuantVectorStore
    raise AttributeError(f"module 'pyturboquant.search' has no attribute {name!r}")
