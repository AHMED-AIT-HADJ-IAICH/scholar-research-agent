from __future__ import annotations

import asyncio
import os

import httpx
from .models import Paper

BASE = "https://api.openalex.org/works"
FIELDS = "id,display_name,publication_year,doi,primary_location,abstract_inverted_index,cited_by_count,authorships,is_retracted,open_access,best_oa_location,type"


def decode_abstract(index: dict | None) -> str | None:
    if not index:
        return None
    words = [(position, word) for word, positions in index.items() for position in positions]
    return " ".join(word for _, word in sorted(words))[:10000]


def parse_paper(work: dict, query: str) -> Paper | None:
    title, year, ident = work.get("display_name"), work.get("publication_year"), work.get("id")
    if not title or not year or not ident or work.get("is_retracted"):
        return None
    location = work.get("primary_location") or {}
    oa = work.get("best_oa_location") or {}
    source = location.get("source") or {}
    doi = work.get("doi")
    return Paper(
        id=ident, title=title, year=year, doi=doi, url=doi or ident,
        authors=[a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])][:12],
        venue=source.get("display_name"), abstract=decode_abstract(work.get("abstract_inverted_index")),
        cited_by_count=work.get("cited_by_count") or 0,
        is_oa=(work.get("open_access") or {}).get("is_oa", False),
        pdf_url=oa.get("pdf_url"), queries=[query],
    )


class OpenAlexClient:
    def __init__(self, api_key: str | None = None, transport=None):
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY")
        self.transport = transport

    async def search(self, query: str, min_year: int, limit: int) -> list[Paper]:
        params = {
            "search": query, "filter": f"from_publication_date:{min_year}-01-01,type:article|preprint|review",
            "per_page": min(limit, 100), "select": FIELDS,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        async with httpx.AsyncClient(timeout=25, transport=self.transport) as client:
            for attempt in range(3):
                try:
                    response = await client.get(BASE, params=params)
                    if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                        await asyncio.sleep(0.8 * (2 ** attempt))
                        continue
                    response.raise_for_status()
                    return [paper for item in response.json().get("results", []) if (paper := parse_paper(item, query))]
                except (httpx.TimeoutException, httpx.ConnectError):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(0.8 * (2 ** attempt))
        return []
