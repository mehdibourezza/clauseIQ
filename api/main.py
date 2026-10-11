"""
ClauseIQ API entry point.

Run locally (from the repository root):
    uvicorn api.main:app --reload --port 8000
Interactive documentation: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.dependencies.providers import get_session_factory
from api.jobs import fail_interrupted_analyses
from api.v1.routers import contracts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Use the same session factory as the endpoints (tests override it).
    session_factory = app.dependency_overrides.get(get_session_factory, get_session_factory)()
    fail_interrupted_analyses(session_factory)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="ClauseIQ",
        description="Contract review: finds 41 CUAD clause types and returns exact quotes found in the contract.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.include_router(contracts.router)
    return app


app = create_app()
