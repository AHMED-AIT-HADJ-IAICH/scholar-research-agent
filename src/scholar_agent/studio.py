import os

from langchain_openai import ChatOpenAI

from scholar_agent.graph import build_graph
from scholar_agent.openalex import OpenAlexClient


def make_graph():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY in .env before starting Studio")
    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), temperature=0)
    return build_graph(model, OpenAlexClient())


graph = make_graph()
