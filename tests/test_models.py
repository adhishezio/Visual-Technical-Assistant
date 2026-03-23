from __future__ import annotations

from backend.core.models import (
    AnswerWithCitations,
    CacheKey,
    ComponentIdentification,
    generate_chunk_id,
)


def test_cache_key_normalizes_same_component_consistently() -> None:
    lower_case = ComponentIdentification(
        manufacturer="siemens ag",
        model_number="6es7 214-1ag40-0xb0",
        part_number="6es7214-1ag40-0xb0",
    )
    mixed_case = ComponentIdentification(
        manufacturer="Siemens AG",
        model_number="6ES7 214-1AG40-0XB0",
        part_number="6ES7214-1AG40-0XB0",
    )

    assert CacheKey.from_identification(lower_case) == CacheKey.from_identification(
        mixed_case
    )


def test_answer_with_citations_updates_flag_from_citations() -> None:
    answer = AnswerWithCitations(answer_text="No answer yet.")

    assert answer.has_citations is False
    assert answer.citations == []


def test_component_serial_is_populated_from_identification_fields() -> None:
    identification = ComponentIdentification(
        manufacturer="ABB",
        model_number="S202",
        part_number="2CDS252001R0404",
    )

    assert identification.component_serial == "ABB_S202_2CDS252001R0404"


def test_generate_chunk_id_is_deterministic() -> None:
    cache_key = CacheKey.from_parts("ABB", "S202", "2CDS252001R0404")

    first = generate_chunk_id(cache_key, chunk_index=0, content_hash="abc123")
    second = generate_chunk_id(cache_key, chunk_index=0, content_hash="abc123")
    third = generate_chunk_id(cache_key, chunk_index=1, content_hash="abc123")

    assert first == second
    assert first != third
