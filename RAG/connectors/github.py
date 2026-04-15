"""GitHub connector — fetches repository files and metadata via PyGitHub."""
from __future__ import annotations

from typing import Any

from connectors.base import BaseConnector, ConnectorPayload
from monitoring.logger import get_logger

logger = get_logger(__name__)


class GitHubConnector(BaseConnector):
    """
    Requires: pip install PyGithub
    config keys:
        repo: str            — "owner/repo"
        branch: str          — default "main"
        path_prefix: str     — subdirectory to scope (optional)
        file_extensions: list[str]  — filter by extension (optional)
    """

    connector_id = "github"

    def fetch(self, config: dict[str, Any]) -> list[ConnectorPayload]:
        try:
            from github import Github, GithubException  # type: ignore
        except ImportError as exc:
            raise ImportError("Install PyGithub: pip install PyGithub") from exc

        from config.settings import settings

        g = Github(settings.github_token)
        repo = g.get_repo(config["repo"])
        branch = config.get("branch", "main")
        prefix = config.get("path_prefix", "")
        extensions: list[str] | None = config.get("file_extensions")

        payloads: list[ConnectorPayload] = []

        def _walk(path: str) -> None:
            try:
                contents = repo.get_contents(path, ref=branch)
            except GithubException as exc:
                logger.warning(f"GitHub error at {path}: {exc}")
                return

            if not isinstance(contents, list):
                contents = [contents]

            for item in contents:
                if item.type == "dir":
                    _walk(item.path)
                elif item.type == "file":
                    if extensions and not any(
                        item.path.endswith(ext) for ext in extensions
                    ):
                        continue
                    try:
                        content = item.decoded_content.decode("utf-8", errors="replace")
                    except Exception as exc:
                        logger.warning(f"Could not decode {item.path}: {exc}")
                        continue

                    payloads.append(ConnectorPayload(
                        connector_id=self.connector_id,
                        source_type="github",
                        raw_content=content,
                        uri=item.html_url,
                        metadata={
                            "repo": config["repo"],
                            "branch": branch,
                            "file_path": item.path,
                            "sha": item.sha,
                            "size": item.size,
                        },
                    ))

        _walk(prefix)
        logger.info(f"GitHubConnector fetched {len(payloads)} files from {config['repo']}")
        return payloads

    def stream(self, config: dict[str, Any]):
        for payload in self.fetch(config):
            yield payload
