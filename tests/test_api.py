"""
End-to-end API tests with a fake extractor and a temporary SQLite database.

TestClient runs background tasks before returning, so the analysis is already finished
when the upload call returns: the polling step is immediate here.
"""

from tests.conftest import CONTRACT, FailingExtractor


def _upload(client, filename="supply.txt", content=CONTRACT.encode()):
    return client.post("/api/v1/contracts", files={"file": (filename, content, "text/plain")})


def test_upload_returns_202_then_analysis_is_available(client):
    response = _upload(client)
    assert response.status_code == 202
    assert response.json()["status"] == "processing"

    detail = client.get(f"/api/v1/contracts/{response.json()['id']}").json()
    assert detail["status"] == "done"
    assert detail["model"] == "fake-model"
    assert detail["clauses_found"] == 2   # Document Name + Governing Law; the reworded quote does not count


def test_verified_findings_carry_offsets_into_the_contract_text(client):
    contract_id = _upload(client).json()["id"]
    detail = client.get(f"/api/v1/contracts/{contract_id}").json()

    governing_law = next(f for f in detail["findings"] if f["clause_type"] == "Governing Law")
    assert governing_law["verified"] is True
    excerpt = detail["text"][governing_law["start_char"]:governing_law["end_char"]]
    assert " ".join(excerpt.split()) == governing_law["quote"]


def test_reworded_quote_is_stored_unverified_without_offsets(client):
    contract_id = _upload(client).json()["id"]
    detail = client.get(f"/api/v1/contracts/{contract_id}").json()

    termination = next(f for f in detail["findings"] if f["clause_type"] == "Termination For Convenience")
    assert termination["verified"] is False
    assert termination["start_char"] is None


def test_extraction_failure_is_recorded_on_the_contract(make_client):
    client = make_client(FailingExtractor())
    contract_id = _upload(client).json()["id"]

    detail = client.get(f"/api/v1/contracts/{contract_id}").json()
    assert detail["status"] == "failed"
    assert "credit balance" in detail["error"]


def test_unsupported_file_type_is_rejected(client):
    response = _upload(client, filename="contract.docx")
    assert response.status_code == 422
    assert ".docx" in response.json()["detail"]


def test_empty_document_is_rejected(client):
    assert _upload(client, content=b"   \n ").status_code == 422


def test_unknown_contract_returns_404(client):
    assert client.get("/api/v1/contracts/does-not-exist").status_code == 404


def test_list_returns_most_recent_first(client):
    first = _upload(client, filename="a.txt").json()["id"]
    second = _upload(client, filename="b.txt").json()["id"]
    ids = [c["id"] for c in client.get("/api/v1/contracts").json()]
    assert ids.index(second) < ids.index(first)


def test_startup_marks_interrupted_analyses_as_failed(tmp_path):
    from fastapi.testclient import TestClient

    from api.dependencies.providers import get_session_factory
    from api.jobs import INTERRUPTED_MESSAGE
    from api.main import create_app
    from data.db.models import Contract
    from data.db.session import make_session_factory

    session_factory = make_session_factory(f"sqlite:///{tmp_path / 'restart.db'}")
    with session_factory() as db:
        db.add(Contract(id="stuck", filename="a.txt", text="text", status="processing"))
        db.add(Contract(id="finished", filename="b.txt", text="text", status="done"))
        db.commit()

    app = create_app()
    app.dependency_overrides[get_session_factory] = lambda: session_factory
    with TestClient(app) as client:   # entering the context runs the startup (lifespan) code
        stuck = client.get("/api/v1/contracts/stuck").json()
        finished = client.get("/api/v1/contracts/finished").json()

    assert stuck["status"] == "failed" and stuck["error"] == INTERRUPTED_MESSAGE
    assert finished["status"] == "done"


def test_clause_types_endpoint_lists_the_catalogue(client):
    clause_types = client.get("/api/v1/clause-types").json()
    assert len(clause_types) == 41
    assert {"name", "description"} <= clause_types[0].keys()
