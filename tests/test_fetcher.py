from __future__ import annotations

from backend.services.fetcher import split_text


def test_split_text_returns_single_chunk_for_short_text() -> None:
    chunks = split_text("Rated current 40A.", chunk_size=1200, chunk_overlap=200)

    assert chunks == ["Rated current 40A."]


def test_split_text_applies_overlap_without_empty_chunks() -> None:
    text = " ".join(f"token{i}" for i in range(120))

    chunks = split_text(text, chunk_size=80, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert chunks[0] != chunks[1]
