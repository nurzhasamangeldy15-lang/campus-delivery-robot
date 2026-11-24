"""Matplotlib picture of the campus: belief heat-map, route, robot and candidate rooms."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from .world import WORLD
from .bayes import CANDIDATES


def draw_map(result, world=WORLD, figsize=(6, 6)):
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    vmax = max(result["belief"].values())
    for r in range(world.rows):
        for c in range(world.cols):
            if world.grid[r][c] == "#":
                ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, color="#3b4252"))
                continue
            p = result["belief"].get((r, c), 0)
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor=plt.cm.Blues(0.08 + 0.8 * p / vmax),
                                   edgecolor="white"))
            if p >= 0.05:
                ax.text(c, r + .3, f"{p:.0%}", ha="center", va="center", fontsize=8, color="#12203A")
    if result["path"]:
        ax.plot([c for _, c in result["path"]], [r for r, _ in result["path"]], color="#2A9D8F", lw=3,
                marker="o", ms=4)
    for name, (r, c) in CANDIDATES.items():
        hit = name == result["target_name"]
        ax.text(c, r - .1, name, ha="center", va="center", fontsize=18, fontweight="bold",
                color="#F2A541" if hit else "#6b7280")
    br, bc = result["best_cell"]
    ax.plot(bc, br, "o", ms=18, mfc="#E4572E", mec="white", mew=2)
    ax.text(bc, br, "R", ha="center", va="center", color="white", fontweight="bold")
    ax.set_xlim(-.5, world.cols - .5); ax.set_ylim(world.rows - .5, -.5)
    ax.set_xticks(range(world.cols)); ax.set_yticks(range(world.rows)); ax.set_aspect("equal")
    ax.set_title("Belief about robot position (blue), route (teal), rooms A/B/C", fontsize=10)
    fig.tight_layout()
    return fig
