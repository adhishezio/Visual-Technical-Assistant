from __future__ import annotations

from backend.core.models import AnswerWithCitations, CacheKey, ComponentIdentification



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

    assert CacheKey.from_identification(lower_case) == CacheKey.from_identification(mixed_case)



def test_answer_with_citations_updates_flag_from_citations() -> None:
    answer = AnswerWithCitations(answer_text="No answer yet.")

    assert answer.has_citations is False
    assert answer.citations == []
