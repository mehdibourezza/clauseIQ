"""
ClauseIQ API entry point.

Run locally (from the repository root):
    uvicorn api.main:app --reload --port 8000
Interactive documentation: http://localhost:8000/docs
"""

from fastapi import FastAPI

from api.v1.routers import contracts


def create_app() -> FastAPI:
    app = FastAPI(
        title="ClauseIQ",
        description="Contract review: finds 41 CUAD clause types and returns verified verbatim excerpts.",
        version="0.2.0",
    )
    app.include_router(contracts.router)
    return app


app = create_app()
