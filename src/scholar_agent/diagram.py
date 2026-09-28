from pathlib import Path


POSITIONS = {
    "__start__": (1.5, 9),
    "plan": (1.5, 8),
    "review": (1.5, 7),
    "search": (1.5, 6),
    "collect": (1.5, 5),
    "analyze": (1.5, 4),
    "choose": (1.5, 3),
    "synthesize": (1.5, 2),
    "finish": (1.5, 1),
    "__end__": (1.5, 0),
}
LABELS = {
    "__start__": "START",
    "plan": "Plan search terms",
    "review": "Researcher review",
    "search": "Search OpenAlex  ·  parallel",
    "collect": "Merge, rank and deduplicate",
    "analyze": "Screen abstracts  ·  parallel",
    "choose": "Select relevant papers",
    "synthesize": "Synthesize evidence",
    "finish": "Build report",
    "__end__": "END",
}


def save_diagram(graph, directory: Path) -> tuple[Path, Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError as exc:
        raise RuntimeError("Install diagram support with pip install -e '.[viz]'") from exc

    topology = graph.get_graph()
    if set(topology.nodes) != set(POSITIONS):
        raise ValueError("Update diagram positions for the current graph nodes")
    directory.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 13), dpi=160)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")

    for edge in topology.edges:
        src, dst = POSITIONS[edge.source], POSITIONS[edge.target]
        skip = src[1] - dst[1] > 1
        lanes = {("review", "__end__"): 3.25,
                 ("collect", "finish"): 3.75,
                 ("choose", "finish"): 4.25}
        x = lanes.get((edge.source, edge.target), 1.5)
        if skip:
            ax.plot([src[0] + 1.12, x, x, dst[0] + 1.12],
                    [src[1], src[1], dst[1], dst[1]],
                    color="#9aa8b6", linewidth=1.5, linestyle="--", zorder=1)
            arrow = FancyArrowPatch((dst[0] + 1.12, dst[1]), (dst[0] + 0.99, dst[1]),
                                    arrowstyle="-|>", mutation_scale=12, color="#9aa8b6", linewidth=1.3)
        else:
            arrow = FancyArrowPatch((src[0], src[1] - 0.28), (dst[0], dst[1] + 0.28),
                                    arrowstyle="-|>", mutation_scale=14, color="#667d98", linewidth=1.6)
        ax.add_patch(arrow)

    for name, (x, y) in POSITIONS.items():
        special = name in ("review", "search", "analyze")
        terminal = name in ("__start__", "__end__")
        color = "#e4f3ed" if special else "#e8eef9" if not terminal else "#dce3ec"
        rect = FancyBboxPatch((x - 1.05, y - 0.29), 2.1, 0.58,
                              boxstyle="round,pad=0.06,rounding_size=0.13",
                              facecolor=color, edgecolor="#7d91aa", linewidth=1.0, zorder=2)
        ax.add_patch(rect)
        ax.text(x, y, LABELS[name], ha="center", va="center", fontsize=10.5,
                fontweight="bold" if terminal or special else "medium", color="#233047", zorder=3)

    ax.text(1.5, 9.7, "SCHOLAR RESEARCH AGENT", ha="center", fontsize=17,
            color="#233047", fontweight="bold")
    ax.text(1.5, 9.4, "Validated search  •  parallel retrieval  •  traceable evidence",
            ha="center", fontsize=10, color="#667d98")
    ax.set_xlim(-0.1, 4.75)
    ax.set_ylim(-0.6, 10.05)
    ax.axis("off")
    fig.tight_layout(pad=0.4)
    png, svg = directory / "agent-graph.png", directory / "agent-graph.svg"
    fig.savefig(png, dpi=160, facecolor=fig.get_facecolor())
    fig.savefig(svg, facecolor=fig.get_facecolor())
    plt.close(fig)
    return png, svg
