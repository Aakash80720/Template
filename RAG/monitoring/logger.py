"""Structured logging + LangSmith tracing setup.

Import get_logger() in every module:
    from monitoring.logger import get_logger
    logger = get_logger(__name__)
"""
import logging
import os

from config.settings import settings


def setup_tracing() -> None:
    """Enable LangSmith tracing if API key is configured."""
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        logging.getLogger(__name__).info(
            f"LangSmith tracing enabled — project: {settings.langsmith_project}"
        )


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '{"time": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "message": "%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# Run once on import
setup_tracing()
