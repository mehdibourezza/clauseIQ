"""
Dependency providers (FastAPI's dependency injection).

Endpoints receive the database session factory and the extraction engine through
Depends(), never by importing them. Tests swap both through app.dependency_overrides:
a temporary SQLite file and a fake extractor, so no test ever calls the paid API.
"""

from functools import lru_cache

import anthropic
from sqlalchemy.orm import Session, sessionmaker

from configs.config import settings
from core.extraction import ClaudeExtractor, ClauseExtractor
from data.db.session import make_session_factory


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return make_session_factory(settings.review_db_url)


@lru_cache
def get_extractor() -> ClauseExtractor:
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
    return ClaudeExtractor(client, model=settings.llm_model, effort=settings.llm_effort)
