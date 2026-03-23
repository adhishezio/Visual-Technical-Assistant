from __future__ import annotations

from backend.core.models import ComponentIdentification, RetrievedChunk

SAFE_NO_CITATIONS_MESSAGE = (
    "I could not answer from official documentation with reliable citations. "
    "Please capture a clearer label image or enter the part number manually."
)

VISION_EXTRACTION_PROMPT = """
You are identifying an industrial or electronic component from a single image.
Return only JSON matching this schema:
{
  "manufacturer": "string or null",
  "model_number": "string or null",
  "part_number": "string or null",
  "component_type": "string or null",
  "visual_description": "string or null",
  "extracted_text": "string",
  "confidence_score": 0.0,
  "part_number_confidence": 0.0
}

Rules:
- Visually identify the component broadly.
- Extract any visible printed text or numbers into extracted_text.
- Prefer exact visible manufacturer, model number, and part number when present.
- Do not guess a part number when it is not visible.
- component_type should be a short noun phrase like "servo drive", "PLC", or "bearing".
- visual_description should describe markings, connectors, housing, and mounting features.
- confidence_score and part_number_confidence must be between 0.0 and 1.0.
""".strip()



def build_chunk_grading_prompt(
    identification: ComponentIdentification | None,
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> str:
    context = format_retrieved_chunks(retrieved_chunks)
    identification_summary = summarize_identification(identification)
    return f"""
You are checking whether retrieved technical documentation is sufficient to answer a user question.
Return only JSON with these fields:
{{
  "sufficient": true,
  "confidence": 0.0,
  "refined_query": "string or null",
  "reasoning": "string"
}}

Question: {question}
Identification: {identification_summary}
Retrieved documentation:
{context}

Rules:
- sufficient=true only if the chunks directly support answering the question.
- If sufficient=false, refined_query should be a better web search query for official docs.
- Prefer manufacturer + part number + document intent in refined_query.
- confidence must be between 0.0 and 1.0.
""".strip()



def build_answer_generation_prompt(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    context = format_retrieved_chunks(retrieved_chunks)
    return f"""
You are answering a technical question using only official documentation excerpts.
Return only JSON with these fields:
{{
  "answer_text": "string",
  "confidence": 0.0,
  "citation_indexes": [0]
}}

Question: {question}
Documentation excerpts:
{context}

Rules:
- Base the answer only on the excerpts above.
- citation_indexes must contain the zero-based excerpt indexes that directly support the answer.
- If the excerpts do not support an answer, answer_text must say that the documentation is insufficient and citation_indexes must be empty.
- When answering questions about power, voltage, or current, explicitly distinguish between the device's direct input specification and the power adapter or supply specification, state which one you are citing, and do not present an adapter specification as if it were the device input.
- confidence must be between 0.0 and 1.0.
""".strip()



def format_retrieved_chunks(retrieved_chunks: list[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return "No retrieved chunks."

    lines: list[str] = []
    for index, retrieved_chunk in enumerate(retrieved_chunks):
        citation_bits = [retrieved_chunk.chunk.metadata.source_url]
        if retrieved_chunk.chunk.page_number is not None:
            citation_bits.append(f"page {retrieved_chunk.chunk.page_number}")
        elif retrieved_chunk.chunk.section_title:
            citation_bits.append(retrieved_chunk.chunk.section_title)
        citation_label = " | ".join(citation_bits)
        lines.append(
            f"[{index}] {citation_label}\n{retrieved_chunk.chunk.chunk_text}"
        )
    return "\n\n".join(lines)



def summarize_identification(identification: ComponentIdentification | None) -> str:
    if identification is None:
        return "None"
    return (
        f"manufacturer={identification.manufacturer!r}, "
        f"model_number={identification.model_number!r}, "
        f"part_number={identification.part_number!r}, "
        f"component_type={identification.component_type!r}"
    )



