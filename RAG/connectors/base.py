"""Abstract base connector — all connectors implement this interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ConnectorPayload(BaseModel):
    """Normalised payload produced by any connector."""
    connector_id: str
    source_type: str                 # "filesystem" | "github" | "mcp" | …
    raw_content: str
    metadata: dict[str, Any] = {}
    mime_type: str = "text/plain"
    uri: str = ""                    # original URI / file path / URL


class BaseConnector(ABC):
    """All connectors must implement `fetch()` and `stream()`."""

    connector_id: str = "base"

    @abstractmethod
    def fetch(self, config: dict[str, Any]) -> list[ConnectorPayload]:
        """Fetch all documents/files matching config. Returns list of payloads."""
        ...

    @abstractmethod
    def stream(self, config: dict[str, Any]):
        """Yield payloads one at a time (for large sources)."""
        ...
