"""MCP (Model Context Protocol) bridge connector.

Translates MCP tool-call responses into ConnectorPayload objects so the
ingestion pipeline can treat MCP servers as first-class data sources.

Usage pattern:
    connector = MCPBridgeConnector()
    payloads  = connector.fetch({
        "tool_name": "read_file",
        "tool_input": {"path": "/docs/spec.md"},
    })
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from config.settings import settings
from connectors.base import BaseConnector, ConnectorPayload
from monitoring.logger import get_logger

logger = get_logger(__name__)


class MCPBridgeConnector(BaseConnector):
    """
    Sends a JSON-RPC 2.0 tool-call to an MCP server and returns the result
    as normalised ConnectorPayload objects.

    config keys:
        tool_name: str         — MCP tool to invoke
        tool_input: dict       — arguments for the tool
        server_url: str        — override settings.mcp_server_url (optional)
        auth_token: str        — override settings.mcp_auth_token (optional)
    """

    connector_id = "mcp"

    def _call_tool(
        self,
        server_url: str,
        auth_token: str | None,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> Any:
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": tool_input},
        }

        response = httpx.post(server_url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()

        if "error" in result:
            raise RuntimeError(f"MCP error: {result['error']}")

        return result.get("result", {})

    def fetch(self, config: dict[str, Any]) -> list[ConnectorPayload]:
        server_url = config.get("server_url") or settings.mcp_server_url
        auth_token = config.get("auth_token") or settings.mcp_auth_token

        if not server_url:
            raise ValueError("MCP server URL not configured (MCP_SERVER_URL or config.server_url)")

        tool_name: str = config["tool_name"]
        tool_input: dict = config.get("tool_input", {})

        raw_result = self._call_tool(server_url, auth_token, tool_name, tool_input)

        # MCP tool results can be content blocks (text/resource) or raw dicts
        payloads: list[ConnectorPayload] = []

        content_blocks = raw_result if isinstance(raw_result, list) else [raw_result]

        for block in content_blocks:
            if isinstance(block, str):
                text = block
                meta: dict = {}
            elif isinstance(block, dict):
                text = block.get("text") or json.dumps(block)
                meta = {k: v for k, v in block.items() if k != "text"}
            else:
                continue

            payloads.append(ConnectorPayload(
                connector_id=self.connector_id,
                source_type="mcp",
                raw_content=text,
                uri=f"{server_url}/{tool_name}",
                metadata={
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    **meta,
                },
            ))

        logger.info(f"MCPBridgeConnector got {len(payloads)} blocks from tool '{tool_name}'")
        return payloads

    def stream(self, config: dict[str, Any]):
        for payload in self.fetch(config):
            yield payload


# ---------------------------------------------------------------------------
# Registry — add new connectors here
# ---------------------------------------------------------------------------

from connectors.filesystem import FilesystemConnector
from connectors.github import GitHubConnector

CONNECTOR_REGISTRY: dict[str, BaseConnector] = {
    "filesystem": FilesystemConnector(),
    "github": GitHubConnector(),
    "mcp": MCPBridgeConnector(),
}
