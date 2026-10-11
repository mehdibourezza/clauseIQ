"""
Background analysis job, run after the upload request has already returned 202.

In an Azure deployment this function would be the body of a queue-triggered Azure Function;
here it runs in FastAPI's background task pool (lost if the server restarts mid-analysis).
"""

import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session, sessionmaker

from core.analysis import analyze_contract
from core.extraction import ClauseExtractor, ExtractionError
from data.db.models import ClauseFinding, Contract

logger = logging.getLogger(__name__)


def run_analysis(contract_id: str, session_factory: sessionmaker[Session], extractor: ClauseExtractor) -> None:
    with session_factory() as db:
        contract = db.get(Contract, contract_id)
        if contract is None:
            return

        try:
            result = analyze_contract(contract.text, extractor)
        except ExtractionError as exc:
            contract.status, contract.error = "failed", str(exc)
        except Exception:
            # Nobody awaits a background job: an unrecorded failure would leave the
            # contract in "processing" forever, so every error must end up in the row.
            logger.exception("Analysis failed for contract %s", contract_id)
            contract.status, contract.error = "failed", "Unexpected error during analysis (see server logs)."
        else:
            contract.findings = [
                ClauseFinding(
                    clause_type=f.clause_type,
                    quote=f.quote,
                    start_char=f.span.start if f.span else None,
                    end_char=f.span.end if f.span else None,
                    verified=f.verified,
                )
                for f in result.findings
            ]
            contract.status = "done"
            contract.model = result.model
            contract.input_tokens = result.input_tokens
            contract.output_tokens = result.output_tokens

        contract.completed_at = datetime.now(timezone.utc)
        db.commit()
