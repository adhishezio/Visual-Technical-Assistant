from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel

from backend.core.config import Settings, get_settings

ModelT = TypeVar("ModelT", bound=BaseModel)


class GeminiServiceError(RuntimeError):
    """Raised when Gemini cannot be called or parsed."""



def build_gemini_client(settings: Settings | None = None) -> Any:
    resolved_settings = settings or get_settings()

    try:
        from google import genai
    except ImportError as exc:
        raise GeminiServiceError("google-genai is required to call Gemini.") from exc

    if resolved_settings.google_api_key:
        return genai.Client(api_key=resolved_settings.google_api_key)

    if resolved_settings.google_cloud_project:
        return genai.Client(
            vertexai=True,
            project=resolved_settings.google_cloud_project,
            location=resolved_settings.google_cloud_location,
        )

    raise GeminiServiceError(
        "Gemini credentials are not configured. Set GOOGLE_API_KEY for local "
        "development or GOOGLE_CLOUD_PROJECT for Vertex AI."
    )



def generate_structured_content(
    prompt: str,
    response_schema: type[ModelT],
    settings: Settings | None = None,
    image_bytes: bytes | None = None,
    mime_type: str | None = None,
) -> ModelT:
    resolved_settings = settings or get_settings()

    try:
        from google.genai import types
    except ImportError as exc:
        raise GeminiServiceError("google-genai is required to call Gemini.") from exc

    client = build_gemini_client(resolved_settings)
    contents: list[object] = [prompt]
    if image_bytes is not None:
        if not mime_type:
            raise GeminiServiceError("mime_type is required when image_bytes are provided.")
        contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    try:
        response = client.models.generate_content(
            model=resolved_settings.gemini_model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_json_schema=_with_property_ordering(
                    response_schema.model_json_schema()
                ),
            ),
        )
    except Exception as exc:
        raise GeminiServiceError("Gemini request failed.") from exc

    parsed = getattr(response, "parsed", None)
    try:
        if parsed is not None:
            if isinstance(parsed, response_schema):
                return parsed
            return response_schema.model_validate(parsed)

        payload = getattr(response, "text", None)
        if not payload:
            payload = _collect_response_text(response)
        if not payload:
            raise GeminiServiceError("Gemini returned an empty response payload.")

        try:
            return response_schema.model_validate_json(payload)
        except ValueError:
            return response_schema.model_validate(json.loads(payload))
    except GeminiServiceError:
        raise
    except Exception as exc:
        raise GeminiServiceError("Gemini returned invalid structured output.") from exc



def _with_property_ordering(schema: dict[str, Any]) -> dict[str, Any]:
    ordered = dict(schema)
    properties = ordered.get("properties")
    if isinstance(properties, dict):
        ordered["propertyOrdering"] = list(properties.keys())
        ordered["properties"] = {
            key: _with_property_ordering(value) if isinstance(value, dict) else value
            for key, value in properties.items()
        }
    if isinstance(ordered.get("items"), dict):
        ordered["items"] = _with_property_ordering(ordered["items"])
    return ordered



def _collect_response_text(response: object) -> str:
    candidates = getattr(response, "candidates", []) or []
    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                parts.append(part_text)
    return "\n".join(parts)
