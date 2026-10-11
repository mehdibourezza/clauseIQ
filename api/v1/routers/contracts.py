"""
Contract review endpoints.

Upload follows the "202 Accepted + polling" pattern: the analysis takes minutes, so the
upload returns immediately with the contract id and the client polls GET /contracts/{id}
until status is "done" or "failed".
"""

from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from api.dependencies.providers import get_extractor, get_session_factory
from api.jobs import run_analysis
from api.schemas.contracts import ClauseTypeOut, ContractDetail, ContractSummary, FindingOut
from core.clause_types import CLAUSE_TYPES
from core.extraction import ClauseExtractor
from core.text_extraction import DocumentError, extract_text
from data.db.models import Contract

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

router = APIRouter(prefix="/api/v1")


def _as_utc(value: datetime | None) -> datetime | None:
    # SQLite drops timezone info; values are stored in UTC, so restore it before serialising.
    return value.replace(tzinfo=timezone.utc) if value and value.tzinfo is None else value


def _summary(contract: Contract) -> ContractSummary:
    return ContractSummary(
        id=contract.id,
        filename=contract.filename,
        status=contract.status,
        created_at=_as_utc(contract.created_at),
        completed_at=_as_utc(contract.completed_at),
        clauses_found=len({f.clause_type for f in contract.findings if f.verified}),
    )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/clause-types", response_model=list[ClauseTypeOut])
def list_clause_types() -> list[ClauseTypeOut]:
    return [ClauseTypeOut(name=c.name, description=c.description) for c in CLAUSE_TYPES]


@router.post("/contracts", status_code=202, response_model=ContractSummary)
def upload_contract(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
    extractor: ClauseExtractor = Depends(get_extractor),
) -> ContractSummary:
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File larger than 10 MB.")
    try:
        text = extract_text(file.filename or "", data)
    except DocumentError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    with session_factory() as db:
        contract = Contract(filename=file.filename, text=text)
        db.add(contract)
        db.commit()
        summary = _summary(contract)

    background_tasks.add_task(run_analysis, contract.id, session_factory, extractor)
    return summary


@router.get("/contracts", response_model=list[ContractSummary])
def list_contracts(
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> list[ContractSummary]:
    with session_factory() as db:
        contracts = db.scalars(
            select(Contract).options(selectinload(Contract.findings)).order_by(Contract.created_at.desc())
        ).all()
        return [_summary(c) for c in contracts]


@router.get("/contracts/{contract_id}", response_model=ContractDetail)
def get_contract(
    contract_id: str,
    session_factory: sessionmaker[Session] = Depends(get_session_factory),
) -> ContractDetail:
    with session_factory() as db:
        contract = db.get(Contract, contract_id, options=[selectinload(Contract.findings)])
        if contract is None:
            raise HTTPException(status_code=404, detail="Contract not found.")
        return ContractDetail(
            **_summary(contract).model_dump(),
            text=contract.text,
            error=contract.error,
            model=contract.model,
            input_tokens=contract.input_tokens,
            output_tokens=contract.output_tokens,
            findings=[
                FindingOut(
                    clause_type=f.clause_type,
                    quote=f.quote,
                    start_char=f.start_char,
                    end_char=f.end_char,
                    verified=f.verified,
                )
                for f in contract.findings
            ],
        )
