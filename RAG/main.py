"""Entry point — run the knowledge-graph ingestion pipeline from the CLI.

Examples:
    # Ingest a local directory
    python main.py filesystem --path ./docs

    # Ingest a GitHub repo
    python main.py github --repo owner/repo --branch main

    # Ingest via MCP tool
    python main.py mcp --tool read_file --input '{"path": "/spec.md"}'
"""
from __future__ import annotations

import argparse
import json
import sys

from pipelines.ingestion.graph import run_ingestion
from guards.input_guard import validate_request
from guards.output_guard import validate_result
from monitoring.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Graph ingestion pipeline")
    sub = parser.add_subparsers(dest="source_type", required=True)

    # filesystem subcommand
    fs = sub.add_parser("filesystem", help="Ingest local files")
    fs.add_argument("--path", required=True, help="Root directory to walk")
    fs.add_argument("--glob", default="**/*", help="Glob pattern (default: **/*)")

    # github subcommand
    gh = sub.add_parser("github", help="Ingest a GitHub repository")
    gh.add_argument("--repo", required=True, help="owner/repo")
    gh.add_argument("--branch", default="main")
    gh.add_argument("--prefix", default="", help="Subdirectory prefix")

    # mcp subcommand
    mcp = sub.add_parser("mcp", help="Ingest via MCP tool call")
    mcp.add_argument("--tool", required=True, help="MCP tool name")
    mcp.add_argument("--input", default="{}", help="JSON tool input")

    # common options
    for p in (fs, gh, mcp):
        p.add_argument("--thread-id", default="default", help="LangGraph thread ID")

    args = parser.parse_args()

    # Build source_config per connector
    if args.source_type == "filesystem":
        source_config = {"root_path": args.path, "glob": args.glob}
    elif args.source_type == "github":
        source_config = {"repo": args.repo, "branch": args.branch, "path_prefix": args.prefix}
    elif args.source_type == "mcp":
        source_config = {"tool_name": args.tool, "tool_input": json.loads(args.input)}
    else:
        logger.error(f"Unknown source type: {args.source_type}")
        sys.exit(1)

    # Input guard
    try:
        validate_request({
            "source_type": args.source_type,
            "source_config": source_config,
        })
    except Exception as exc:
        logger.error(f"Input validation failed: {exc}")
        sys.exit(1)

    # Run pipeline
    logger.info(f"Starting ingestion: source_type={args.source_type}")
    try:
        final_state = run_ingestion(
            source_type=args.source_type,
            source_config=source_config,
            thread_id=args.thread_id,
        )
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}")
        sys.exit(1)

    summary = final_state.get("ingestion_summary", {})
    errors = final_state.get("errors", [])

    # Output guard
    result = validate_result({
        "success": not errors,
        "entities_created": summary.get("entities_stored", 0),
        "relationships_created": summary.get("relationships_stored", 0),
        "errors": errors,
        "summary": summary,
    })

    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
