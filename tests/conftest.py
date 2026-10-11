import pytest
from fastapi.testclient import TestClient

from api.dependencies.providers import get_extractor, get_session_factory
from api.main import create_app
from core.clause_types import CLAUSE_NAMES
from core.extraction import ExtractionError, ExtractionResult
from data.db.session import make_session_factory

CONTRACT = (
    "SUPPLY AGREEMENT\n"
    "This Agreement shall be governed by the laws of\nthe State of Delaware.\n"
    "Either party may terminate this Agreement for convenience upon 30 days notice."
)


class FakeExtractor:
    """Returns canned excerpts: one real quote, one reworded (must fail verification)."""

    def extract(self, contract_text: str) -> ExtractionResult:
        excerpts = {name: [] for name in CLAUSE_NAMES}
        excerpts["Document Name"] = ["SUPPLY AGREEMENT"]
        excerpts["Governing Law"] = ["This Agreement shall be governed by the laws of the State of Delaware."]
        excerpts["Termination For Convenience"] = ["Either party can end the Agreement at any time."]
        return ExtractionResult(excerpts, model="fake-model", input_tokens=100, output_tokens=20)


class FailingExtractor:
    def extract(self, contract_text: str) -> ExtractionResult:
        raise ExtractionError("LLM API error (400): credit balance too low")


@pytest.fixture
def make_client(tmp_path):
    def _make(extractor) -> TestClient:
        app = create_app()
        session_factory = make_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
        app.dependency_overrides[get_session_factory] = lambda: session_factory
        app.dependency_overrides[get_extractor] = lambda: extractor
        return TestClient(app)

    return _make


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client(FakeExtractor())
