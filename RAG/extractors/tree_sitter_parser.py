"""tree-sitter code parser — extracts structural entities from source files.

Requires: pip install tree-sitter tree-sitter-python tree-sitter-javascript
          tree-sitter-typescript tree-sitter-go tree-sitter-rust
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from monitoring.logger import get_logger

logger = get_logger(__name__)

# Map file extension → tree-sitter language module name
EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".rb": "ruby",
}

# Query templates per language for extracting top-level declarations
# Extend these with language-specific node types as needed
QUERIES: dict[str, dict[str, str]] = {
    "python": {
        "functions": "(function_definition name: (identifier) @name)",
        "classes": "(class_definition name: (identifier) @name)",
        "imports": "(import_statement) @import",
        "from_imports": "(import_from_statement) @import",
    },
    "javascript": {
        "functions": "[  (function_declaration name: (identifier) @name)  (arrow_function) @arrow]",
        "classes": "(class_declaration name: (identifier) @name)",
        "imports": "(import_statement) @import",
    },
    "typescript": {
        "functions": "[  (function_declaration name: (identifier) @name)  (method_definition name: (property_identifier) @name)]",
        "classes": "(class_declaration name: (identifier) @name)",
        "interfaces": "(interface_declaration name: (type_identifier) @name)",
        "imports": "(import_statement) @import",
    },
    "go": {
        "functions": "(function_declaration name: (identifier) @name)",
        "types": "(type_declaration) @type",
        "imports": "(import_declaration) @import",
    },
    "rust": {
        "functions": "(function_item name: (identifier) @name)",
        "structs": "(struct_item name: (type_identifier) @name)",
        "enums": "(enum_item name: (type_identifier) @name)",
        "use": "(use_declaration) @import",
    },
}


class ParsedCodeEntity(BaseModel):
    file_path: str
    language: str
    functions: list[str] = []
    classes: list[str] = []
    interfaces: list[str] = []
    imports: list[str] = []
    structs: list[str] = []
    enums: list[str] = []
    raw_declarations: list[dict[str, Any]] = []


def _load_language(lang_name: str):
    """Lazy-load a tree-sitter language binding."""
    try:
        import importlib
        # New-style bindings (tree-sitter >= 0.22): tree_sitter_<lang>
        module_name = f"tree_sitter_{lang_name.replace('-', '_')}"
        mod = importlib.import_module(module_name)
        from tree_sitter import Language
        return Language(mod.language())
    except (ImportError, AttributeError) as exc:
        logger.warning(f"tree-sitter language '{lang_name}' not available: {exc}")
        return None


def parse_code(content: str, file_path: str) -> ParsedCodeEntity | None:
    """Parse source code and return structured code entity."""
    ext = Path(file_path).suffix.lower()
    lang_name = EXTENSION_LANGUAGE_MAP.get(ext)

    if not lang_name:
        logger.debug(f"No tree-sitter parser for extension '{ext}'")
        return None

    language = _load_language(lang_name)
    if language is None:
        return None

    try:
        from tree_sitter import Parser
        parser = Parser(language)
        tree = parser.parse(bytes(content, "utf-8"))
    except Exception as exc:
        logger.warning(f"tree-sitter parse failed for {file_path}: {exc}")
        return None

    queries = QUERIES.get(lang_name, {})
    result = ParsedCodeEntity(file_path=file_path, language=lang_name)

    for query_name, query_string in queries.items():
        try:
            q = language.query(query_string)
            captures = q.captures(tree.root_node)

            names: list[str] = []
            for node, capture_name in captures:
                if capture_name == "name":
                    names.append(node.text.decode("utf-8"))
                elif capture_name in ("import", "arrow"):
                    # Capture full import text
                    names.append(node.text.decode("utf-8")[:120])  # truncate

            if "function" in query_name:
                result.functions.extend(names)
            elif "class" in query_name:
                result.classes.extend(names)
            elif "interface" in query_name:
                result.interfaces.extend(names)
            elif "import" in query_name or "use" in query_name:
                result.imports.extend(names)
            elif "struct" in query_name:
                result.structs.extend(names)
            elif "enum" in query_name:
                result.enums.extend(names)

        except Exception as exc:
            logger.debug(f"Query '{query_name}' failed for {file_path}: {exc}")

    # Deduplicate
    result.functions = list(dict.fromkeys(result.functions))
    result.classes = list(dict.fromkeys(result.classes))
    result.imports = list(dict.fromkeys(result.imports))

    logger.info(
        f"Parsed {file_path}: {len(result.functions)} functions, "
        f"{len(result.classes)} classes, {len(result.imports)} imports"
    )
    return result
