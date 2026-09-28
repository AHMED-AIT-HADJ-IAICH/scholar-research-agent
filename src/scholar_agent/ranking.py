from __future__ import annotations

import math
import re
from datetime import date
from .models import Paper

STOP = set("the and for with from using use study studies review systematic a an of in on to by et de des du la le les une un pour sur dans avec et ou".split())


def tokens(text: str) -> set[str]:
    return {s for s in re.findall(r"[^\W_]+", text.casefold()) if len(s) > 2 and s not in STOP}


def rank(papers: list[Paper], concepts: list[str], exclusions: list[str]) -> list[Paper]:
    current_year = date.today().year
    concept_tokens = [tokens(c) for c in concepts if tokens(c)]
    exclusion_tokens = [tokens(e) for e in exclusions if tokens(e)]
    ranked = []
    for paper in papers:
        title, abstract = tokens(paper.title), tokens(paper.abstract or "")
        text = title | abstract
        if any(phrase <= text for phrase in exclusion_tokens):
            continue
        coverage = sum(1 for c in concept_tokens if c & text) / max(len(concept_tokens), 1)
        title_coverage = sum(1 for c in concept_tokens if c & title) / max(len(concept_tokens), 1)
        relevance = 0.7 * coverage + 0.3 * title_coverage
        citations = min(math.log1p(paper.cited_by_count) / math.log1p(500), 1)
        age = max(0, current_year - paper.year)
        annual = min(math.log1p(paper.cited_by_count / (age + 1)) / math.log1p(40), 1)
        recency = math.exp(-age / 8)
        score = 0.55 * relevance + 0.15 * citations + 0.15 * annual + 0.15 * recency
        paper.score = round(score, 4)
        paper.score_breakdown = {"relevance": round(relevance, 3), "citations": round(citations, 3), "citations_per_year": round(annual, 3), "recency": round(recency, 3)}
        ranked.append(paper)
    return sorted(ranked, key=lambda p: (-p.score, p.id))


def deduplicate(papers: list[Paper]) -> list[Paper]:
    items: dict[str, Paper] = {}
    for paper in papers:
        key = (paper.doi or "").casefold().strip() or re.sub(r"\W+", "", paper.title.casefold())
        if key in items:
            existing = items[key]
            existing.queries = sorted(set(existing.queries + paper.queries))
            if not existing.abstract and paper.abstract:
                existing.abstract = paper.abstract
            existing.cited_by_count = max(existing.cited_by_count, paper.cited_by_count)
        else:
            items[key] = paper
    return list(items.values())
