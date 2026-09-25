from qualification.scoring import company_evidence_investigator as investigator


def test_missing_quote_fragment_gets_bounded_real_page_excerpt() -> None:
    page = (
        "Latent, the clinical-AI company accelerating access to life-saving "
        "medications, has raised an $80M Series A co-led by Spark Capital and "
        "Transformation Capital. "
        + "Verified article body. " * 200
    )
    invented_quote = (
        "Latent secured a large early-stage investment to expand patient access."
    )

    context = investigator._source_context_for_quote(invented_quote, page)

    assert not investigator._quote_occurs(invented_quote, page)
    assert context == page[:(
        investigator.REJECTED_QUOTE_CONTEXT_BEFORE_CHARACTERS
        + investigator.REJECTED_QUOTE_CONTEXT_AFTER_CHARACTERS
    )]
    assert "Latent" in context
    assert "Series A" in context


def test_exact_fragment_keeps_local_context_instead_of_page_prefix() -> None:
    prefix = "Unrelated introduction. " * 100
    exact_fragment = "Metriport has raised $26 million in new funding."
    page = prefix + exact_fragment + " Evidence after the quoted sentence."

    context = investigator._source_context_for_quote(
        exact_fragment + " Invented continuation.", page
    )

    assert exact_fragment in context
    assert context != page[:(
        investigator.REJECTED_QUOTE_CONTEXT_BEFORE_CHARACTERS
        + investigator.REJECTED_QUOTE_CONTEXT_AFTER_CHARACTERS
    )]
