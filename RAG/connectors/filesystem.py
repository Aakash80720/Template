"""Local filesystem connector — walks a directory and yields file payloads."""
from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorPayload
from monitoring.logger import get_logger

logger = get_logger(__name__)

# File extensions treated as code
CODE_EXTENSIONS = {
    ".py", ".ts", ".js", ".tsx", ".jsx", ".go", ".rs",
    ".java", ".cpp", ".c", ".h", ".cs", ".rb", ".swift",
}


class FilesystemConnector(BaseConnector):
    connector_id = "filesystem"

    def fetch(self, config: dict[str, Any]) -> list[ConnectorPayload]:
        """
        config keys:
            root_path: str          — directory to walk
            glob: str               — e.g. "**/*" (default)
            max_file_size_kb: int   — skip files larger than this (default 512)
        """
        root = Path(config["root_path"])
        glob_pattern = config.get("glob", "**/*")
        max_kb = config.get("max_file_size_kb", 512)

        payloads: list[ConnectorPayload] = []
        for path in root.glob(glob_pattern):
            if not path.is_file():
                continue
            if path.stat().st_size > max_kb * 1024:
                logger.info(f"Skipping large file: {path}")
                continue

            mime, _ = mimetypes.guess_type(str(path))
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.warning(f"Could not read {path}: {exc}")
                continue

            payloads.append(ConnectorPayload(
                connector_id=self.connector_id,
                source_type="filesystem",
                raw_content=content,
                mime_type=mime or "text/plain",
                uri=str(path.resolve()),
                metadata={
                    "file_name": path.name,
                    "extension": path.suffix,
                    "is_code": path.suffix in CODE_EXTENSIONS,
                    "relative_path": str(path.relative_to(root)),
                },
            ))

        logger.info(f"FilesystemConnector fetched {len(payloads)} files from {root}")
        return payloads

    def stream(self, config: dict[str, Any]):
        for payload in self.fetch(config):
            yield payload
