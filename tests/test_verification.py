from core.verification import QuoteVerifier

CONTRACT = (
    "ARTICLE 12. GOVERNING LAW\n"
    "This Agreement shall be governed by the laws of\n"
    "the State of  Delaware. The parties’ obligations survive termination — see Section 9."
)


def test_exact_quote_returns_offsets_in_original_text():
    span = QuoteVerifier(CONTRACT).locate("ARTICLE 12. GOVERNING LAW")
    assert span is not None
    assert CONTRACT[span.start:span.end] == "ARTICLE 12. GOVERNING LAW"


def test_quote_matches_across_line_breaks_and_double_spaces():
    quote = "This Agreement shall be governed by the laws of the State of Delaware."
    span = QuoteVerifier(CONTRACT).locate(quote)
    assert span is not None
    assert " ".join(CONTRACT[span.start:span.end].split()) == quote


def test_typographic_punctuation_matches_plain_ascii():
    span = QuoteVerifier(CONTRACT).locate("The parties' obligations survive termination - see Section 9.")
    assert span is not None
    assert CONTRACT[span.end - 1] == "."


def test_paraphrased_quote_is_rejected():
    assert QuoteVerifier(CONTRACT).locate("This contract is governed by Delaware law.") is None


def test_case_change_is_rejected():
    assert QuoteVerifier(CONTRACT).locate("article 12. governing law") is None


def test_locate_all_returns_every_occurrence_and_locate_the_first():
    text = "SUPPLY AGREEMENT\nterms...\nSUPPLY  AGREEMENT"
    verifier = QuoteVerifier(text)
    spans = verifier.locate_all("SUPPLY AGREEMENT")
    assert [text[s.start:s.end].split() for s in spans] == [["SUPPLY", "AGREEMENT"]] * 2
    assert verifier.locate("SUPPLY AGREEMENT") == spans[0]


def test_trivially_short_quote_is_rejected():
    assert QuoteVerifier(CONTRACT).locate("of") is None
    assert QuoteVerifier(CONTRACT).locate("   ") is None
