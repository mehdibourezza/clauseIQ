"""
Architecture rule, checked on every test run: the business core (core/) imports no web
framework, no database layer, no model provider and no other package of this application.
"""

import re
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "core"
FORBIDDEN_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(fastapi|starlette|sqlalchemy|anthropic|api|data|engines|scripts)\b",
    re.MULTILINE,
)


def test_core_imports_no_framework_database_or_provider():
    offenders = {
        path.name: FORBIDDEN_IMPORT.findall(path.read_text(encoding="utf-8"))
        for path in CORE.glob("*.py")
    }
    assert {name: found for name, found in offenders.items() if found} == {}
