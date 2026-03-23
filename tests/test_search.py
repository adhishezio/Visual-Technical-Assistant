from __future__ import annotations

from backend.core.models import ComponentIdentification
from backend.services.search import (
    build_search_queries,
    guess_official_domains,
    official_source_boost,
)


def test_build_search_queries_generates_default_document_hints() -> None:
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S 204 M B 40 UC",
        part_number="2CD274061R0405",
        component_type="Miniature Circuit Breaker",
        confidence_score=1.0,
        raw_ocr_text="ABB S 204 M B 40 UC",
        should_attempt_document_lookup=True,
    )

    queries = build_search_queries(identification)

    assert len(queries) == 3
    assert "official datasheet" in queries[0].query.lower()
    assert "technical manual" in queries[1].query.lower()
    assert "wiring diagram" in queries[2].query.lower()
    assert "2CD274061R0405" in queries[0].query


def test_build_search_queries_uses_refined_query_when_present() -> None:
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S 204 M B 40 UC",
        part_number="2CD274061R0405",
        component_type="Miniature Circuit Breaker",
        confidence_score=1.0,
        raw_ocr_text="ABB S 204 M B 40 UC",
        should_attempt_document_lookup=True,
    )

    queries = build_search_queries(
        identification,
        refined_query="ABB S 204 M B 40 UC rated current datasheet pdf",
    )

    assert len(queries) == 1
    assert queries[0].query == "ABB S 204 M B 40 UC rated current datasheet pdf"


def test_guess_official_domains_strips_corporate_suffixes() -> None:
    assert guess_official_domains("Schneider Electric Ltd") == ["schneider.com"]
    assert guess_official_domains("ABB AG") == ["abb.com"]


def test_official_source_boost_favors_matching_manufacturer_hostname() -> None:
    boost = official_source_boost(
        "https://library.abb.com/public/spec-sheet.pdf",
        "abb",
    )
    no_boost = official_source_boost(
        "https://example.com/spec-sheet.pdf",
        "abb",
    )

    assert boost > no_boost
