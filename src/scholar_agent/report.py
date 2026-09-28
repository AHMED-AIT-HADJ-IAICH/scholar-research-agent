from __future__ import annotations

from .models import Result


def render_report(result: Result) -> str:
    lines = [f"# Literature review: {result.question}", "", f"Publication year: {result.min_year} or later | Requested: {result.requested_count} | Included: {len(result.papers)}", "", "## Search method", "", f"Approved concepts: {', '.join(result.plan.concepts)}", f"Queries: {'; '.join(result.searched_queries)}", f"Unique candidates: {result.candidate_count}. OpenAlex metadata and available abstracts were screened. Citation counts may change. This is not a full-text systematic review.", ""]
    if result.warnings:
        lines += ["## Warnings", ""] + [f"- {warning}" for warning in result.warnings] + [""]
    if result.synthesis:
        lines += ["## Overview", "", result.synthesis.overview, "", "## Themes", ""]
        lines += [f"- {theme}" for theme in result.synthesis.themes]
        lines += ["", "## Research gaps", ""] + [f"- {gap}" for gap in result.synthesis.gaps]
        lines += ["", "## Next questions", ""] + [f"- {q}" for q in result.synthesis.next_questions]
    evidence = {e.paper_id: e for e in result.evidence}
    lines += ["", "## Selected papers", ""]
    for index, paper in enumerate(result.papers, 1):
        ev = evidence.get(paper.id)
        lines += [f"### {index}. {paper.title} ({paper.year})", "", f"{', '.join(paper.authors[:5]) or 'Authors unavailable'} | {paper.venue or 'Venue unavailable'} | {paper.cited_by_count} citations", "", f"[Record]({paper.url})" + (f" | [Open PDF]({paper.pdf_url})" if paper.pdf_url else ""), "", f"Relevance: {ev.relevance if ev else 'unknown'}/3 | Score: {paper.score:.3f} | Open access: {'yes' if paper.is_oa else 'no'}", "", f"Contribution: {ev.contribution if ev else 'Unavailable'}", "", f"Method: {ev.method or 'Not stated in abstract' if ev else 'Unavailable'}", "", f"Limitation: {ev.limitation or 'Not stated in abstract' if ev else 'Unavailable'}", ""]
    return "\n".join(lines)
