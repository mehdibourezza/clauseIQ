from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from data.db.models import Base


def make_session_factory(db_url: str) -> sessionmaker[Session]:
    """
    Create the engine, create missing tables, and return a session factory.

    Tables are created with create_all (no migrations): acceptable for the local prototype,
    to be replaced by Alembic before any shared deployment.
    """
    # SQLite connections are bound to their creating thread by default; FastAPI serves
    # requests and background tasks from a thread pool, so that check must be disabled.
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)
