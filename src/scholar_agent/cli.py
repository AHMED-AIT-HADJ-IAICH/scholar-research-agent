from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .graph import build_graph
from .models import ResearchRequest, Result
from .openalex import OpenAlexClient
from .report import render_report
from .diagram import save_diagram


async def run(args):
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in the environment or .env")
    request = ResearchRequest(question=args.question, min_year=args.min_year, paper_count=args.count, language=args.language)
    graph = build_graph(ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0), OpenAlexClient(), InMemorySaver())
    config = {"configurable": {"thread_id": str(uuid4())}, "max_concurrency": 4}
    await graph.ainvoke({"request": request.model_dump()}, config)
    state = await graph.aget_state(config)
    if not state.interrupts:
        raise RuntimeError("Expected a search-plan review")
    plan = state.values["plan"]
    print("\nProposed concepts:", ", ".join(plan["concepts"]))
    for index, query in enumerate(plan["queries"], 1):
        print(f"  {index}. {query}")
    print(f"Year >= {request.min_year} | Papers: {request.paper_count}")
    choice = input("Approve [y], edit [e], cancel [n]: ").strip().lower()
    if choice == "e":
        raw = input("Queries separated by semicolons: ").strip()
        concepts = input("Concepts separated by semicolons (blank = keep): ").strip()
        decision = {"approved": True, "queries": [q.strip() for q in raw.split(";") if q.strip()]}
        if concepts:
            decision["concepts"] = [c.strip() for c in concepts.split(";") if c.strip()]
    else:
        decision = {"approved": choice == "y"}
    if not decision["approved"]:
        print("Research cancelled.")
        return
    output = await graph.ainvoke(Command(resume=decision), config)
    result = Result.model_validate(output["result"])
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.md").write_text(render_report(result), encoding="utf-8")
    (target / "result.json").write_text(json.dumps(result.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(result.papers)} papers in {target.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Research assistant for scientific papers")
    parser.add_argument("question", nargs="?")
    parser.add_argument("--min-year", type=int)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--language", choices=["French", "English"], default="French")
    parser.add_argument("--output", default="reports/latest")
    parser.add_argument("--graph", action="store_true", help="Save the graph as PNG and SVG")
    args = parser.parse_args()
    if args.graph:
        graph = build_graph(object(), OpenAlexClient(), InMemorySaver())
        png, svg = save_diagram(graph, Path("assets"))
        print(f"Saved {png} and {svg}")
        return
    if not args.question or args.min_year is None:
        parser.error("question and --min-year are required unless --graph is used")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
