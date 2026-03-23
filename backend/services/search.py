from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from backend.core.config import SearchProvider, Settings, get_settings
from backend.core.models import ComponentIdentification, DocumentationCandidate, DocumentType


class SearchServiceError(RuntimeError):
    """Raised when documentation search cannot be completed."""


@dataclass(frozen=True)
class SearchQuery:
    query: str
    document_type: DocumentType


class DocumentationSearchService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def search(
        self,
        identification: ComponentIdentification,
        refined_query: str | None = None,
    ) -> list[DocumentationCandidate]:
        if self.settings.search_provider is SearchProvider.NONE:
            return []
        if self.settings.search_provider is not SearchProvider.TAVILY:
            raise SearchServiceError(
                f"Unsupported search provider: {self.settings.search_provider}"
            )
        if not self.settings.tavily_api_key:
            raise SearchServiceError("TAVILY_API_KEY must be configured for search.")

        try:
            from tavily import TavilyClient
        except ImportError as exc:
            raise SearchServiceError("tavily-python is required for search.") from exc

        client = TavilyClient(api_key=self.settings.tavily_api_key)
        manufacturer_token = normalize_manufacturer_token(identification.manufacturer)
        include_domains = guess_official_domains(identification.manufacturer)
        ranked_candidates: dict[str, DocumentationCandidate] = {}

        for search_query in build_search_queries(identification, refined_query):
            search_kwargs: dict[str, object] = {
                "query": search_query.query,
                "max_results": self.settings.tavily_max_results,
                "include_raw_content": "markdown",
                "search_depth": "advanced",
            }
            if include_domains:
                search_kwargs["include_domains"] = include_domains

            response = client.search(**search_kwargs)
            for result in response.get("results", []):
                url = str(result.get("url") or "").strip()
                if not url:
                    continue
                title = str(result.get("title") or "").strip() or None
                score = float(result.get("score") or 0.0)
                document_type = guess_document_type(
                    url,
                    title or "",
                    fallback=search_query.document_type,
                )
                adjusted_score = score + official_source_boost(url, manufacturer_token)
                candidate = DocumentationCandidate(
                    url=url,
                    title=title,
                    document_type=document_type,
                    score=adjusted_score,
                )
                existing = ranked_candidates.get(url)
                if existing is None or candidate.score > existing.score:
                    ranked_candidates[url] = candidate

        return sorted(
            ranked_candidates.values(),
            key=lambda item: item.score,
            reverse=True,
        )


def build_search_queries(
    identification: ComponentIdentification,
    refined_query: str | None = None,
) -> list[SearchQuery]:
    manufacturer = (identification.manufacturer or "").strip()
    identifier = (
        identification.part_number
        or identification.model_number
        or identification.component_type
        or "component"
    )
    if refined_query:
        return [SearchQuery(query=refined_query, document_type=DocumentType.UNKNOWN)]

    return [
        SearchQuery(
            query=f"{manufacturer} {identifier} official datasheet filetype:pdf".strip(),
            document_type=DocumentType.DATASHEET,
        ),
        SearchQuery(
            query=f"{manufacturer} {identifier} technical manual pdf".strip(),
            document_type=DocumentType.MANUAL,
        ),
        SearchQuery(
            query=f"{manufacturer} {identifier} wiring diagram pdf".strip(),
            document_type=DocumentType.WIRING,
        ),
    ]


def guess_document_type(url: str, title: str, fallback: DocumentType = DocumentType.UNKNOWN) -> DocumentType:
    haystack = f"{url} {title}".lower()
    if "datasheet" in haystack or "spec" in haystack:
        return DocumentType.DATASHEET
    if "manual" in haystack or "user guide" in haystack:
        return DocumentType.MANUAL
    if "wiring" in haystack or "schematic" in haystack or "diagram" in haystack:
        return DocumentType.WIRING
    if "safety" in haystack or "warning" in haystack:
        return DocumentType.SAFETY
    return fallback


def normalize_manufacturer_token(manufacturer: str | None) -> str:
    if not manufacturer:
        return ""
    return "".join(character for character in manufacturer.lower() if character.isalnum())


def guess_official_domains(manufacturer: str | None) -> list[str]:
    if not manufacturer:
        return []
    normalized_parts = [
        part.lower()
        for part in manufacturer.replace("-", " ").split()
        if len(part) > 2 and part.lower() not in {"inc", "corp", "co", "ag", "ltd", "llc"}
    ]
    if not normalized_parts:
        return []
    return [f"{normalized_parts[0]}.com"]


def official_source_boost(url: str, manufacturer_token: str) -> float:
    if not manufacturer_token:
        return 0.0
    hostname = urlparse(url).netloc.lower().replace("www.", "")
    hostname_token = "".join(character for character in hostname if character.isalnum())
    return 0.25 if manufacturer_token and manufacturer_token in hostname_token else 0.0
