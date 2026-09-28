import asyncio
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from scholar_agent.graph import build_graph
from scholar_agent.models import Analysis, Evidence, SearchPlan, Synthesis, Paper


class FakeStructured:
    def __init__(self, schema):
        self.schema = schema

    async def ainvoke(self, messages):
        if self.schema is SearchPlan:
            return SearchPlan(concepts=["predictive maintenance", "language models"], queries=["predictive maintenance language models", "maintenance RAG"], rationale="two angles")
        if self.schema is Analysis:
            import json
            papers = json.loads(messages[1][1])["papers"]
            return Analysis(evidence=[Evidence(paper_id=p["id"], relevance=3, contribution="Relevant result") for p in papers])
        return Synthesis(overview="Evidence [W1]", themes=["Evidence [W1]"], gaps=["Unknown"], next_questions=["Replicate?"])


class FakeLLM:
    def with_structured_output(self, schema):
        return FakeStructured(schema)


class FakeClient:
    async def search(self, query, min_year, limit):
        return [Paper(id=f"W{i}", title=f"Predictive maintenance with language models {i}", year=2024, url=f"https://openalex.org/W{i}", abstract="Predictive maintenance with language models", cited_by_count=i) for i in range(1, 5)]


def test_review_parallel_search_and_report():
    async def run():
        graph = build_graph(FakeLLM(), FakeClient(), InMemorySaver())
        config = {"configurable": {"thread_id": "test"}, "max_concurrency": 4}
        await graph.ainvoke({"request": {"question": "How do language models improve predictive maintenance?", "min_year": 2020, "paper_count": 3}}, config)
        assert (await graph.aget_state(config)).interrupts
        result = await graph.ainvoke(Command(resume={"approved": True}), config)
        assert len(result["result"]["papers"]) == 3
        assert len(result["result"]["searched_queries"]) == 2
        assert result["result"]["candidate_count"] == 4
    asyncio.run(run())
