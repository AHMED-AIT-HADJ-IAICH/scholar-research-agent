from scholar_agent.models import Paper, ResearchRequest
from scholar_agent.openalex import decode_abstract, parse_paper
from scholar_agent.ranking import deduplicate, rank
from scholar_agent.report import render_report
from scholar_agent.models import Result, SearchPlan


def paper(id, title, citations=0, year=2024, doi=None, abstract=None):
    return Paper(id=id, title=title, year=year, cited_by_count=citations, doi=doi, url=id, abstract=abstract)


def test_abstract_order_and_retracted_filter():
    assert decode_abstract({"maintenance": [1], "predictive": [0]}) == "predictive maintenance"
    assert parse_paper({"id": "W1", "display_name": "Test", "publication_year": 2024, "is_retracted": True}, "query") is None


def test_dedup_and_relevance_beats_citations():
    related = paper("A", "Predictive maintenance with language models", 10, doi="https://doi.org/10.1/a")
    duplicate = paper("B", related.title, 30, doi="https://doi.org/10.1/a")
    unrelated = paper("C", "Astronomy in ancient history", 10000)
    unique = deduplicate([related, duplicate, unrelated])
    ranked = rank(unique, ["predictive maintenance", "language models"], [])
    assert len(unique) == 2
    assert ranked[0].id == "A"
    assert ranked[0].cited_by_count == 30


def test_report_has_traceable_paper():
    p = paper("https://openalex.org/W1", "Predictive maintenance", 7)
    result = Result(question="Predictive maintenance research?", plan=SearchPlan(concepts=["predictive", "maintenance"], queries=["predictive maintenance", "maintenance AI"], rationale="test"), requested_count=3, min_year=2020, papers=[p], evidence=[], synthesis=None, searched_queries=["predictive maintenance"])
    report = render_report(result)
    assert "https://openalex.org/W1" in report
    assert "full-text systematic review" in report


def test_request_limits():
    import pytest
    with pytest.raises(ValueError):
        ResearchRequest(question="Short", min_year=2020, paper_count=1)
