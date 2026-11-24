#!/usr/bin/env python3
"""
Campus Delivery Robot - Interactive Web Demo
Built with Gradio for AI Fundamentals Final Defense
Student: Nurzhas Amangeldi (QAIRU, 2026)

Run with:
  python app.py
"""

import ast
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from campus_robot.world import WORLD
from campus_robot.pipeline import run
from campus_robot.visual import draw_map
from campus_robot.bayes import CANDIDATES


def parse_readings(text):
    text = text.strip()
    try:
        val = ast.literal_eval(text)
        if isinstance(val, list) and all(isinstance(x, (tuple, list)) and len(x) == 2 for x in val):
            return [tuple(x) for x in val]
    except Exception:
        pass
    # Fallback comma-separated pairs like "4,3; 4,3; 4,3" or "(4,3), (4,3)"
    pairs = []
    clean = text.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    tokens = [t.strip() for t in clean.split(",") if t.strip()]
    for i in range(0, len(tokens) - 1, 2):
        pairs.append((int(tokens[i]), int(tokens[i+1])))
    if not pairs:
        raise ValueError("Invalid format. Example: [(4, 3), (4, 3), (4, 3)]")
    return pairs


def solve_robot(readings_str, clues):
    try:
        readings = parse_readings(readings_str)
        result = run(readings, clues or [])
        fig = draw_map(result, WORLD, figsize=(6, 6))

        # Format HMM text
        top3 = sorted(result["belief"].items(), key=lambda x: -x[1])[:3]
        top3_str = "\n".join([f"  • Cell {c}: {p:.1%}" for c, p in top3])
        hmm_summary = (
            f"**Belief Cell:** `{result['best_cell']}`\n\n"
            f"**Confidence:** `{result['confidence']:.1%}`\n\n"
            f"**Viterbi Sequence:** `{result['viterbi']}`\n\n"
            f"**Top-3 Beliefs:**\n{top3_str}"
        )

        # Format Bayes text
        bayes_summary = (
            f"**Identified Target:** Classroom **{result['target_name']}** at `{result['target']}`\n\n"
            f"**Target Probability:** `{result['posterior'][result['target_name']]:.1%}`\n\n"
            f"**Posterior Distribution:**\n"
            + "\n".join([f"  • Room {k} ({CANDIDATES[k]}): {v:.1%}" for k, v in result['posterior'].items()])
        )

        # Format A* text
        path_str = " ➔ ".join([f"({r},{c})" for r, c in result["path"]])
        astar_summary = (
            f"**Optimal Steps:** `{len(result['path']) - 1}` steps\n\n"
            f"**Nodes Expanded:** `{result['expanded']}` nodes (vs 15 for BFS)\n\n"
            f"**Planned Route:**\n`{path_str}`"
        )

        # Format MDP text
        action = result["action"]
        badge = "🟢 **MOVE**" if action == "Move" else "🟡 **RELOCALIZE**"
        mdp_summary = (
            f"### Recommended Action: {badge}\n\n"
            f"**HMM Confidence P(Localized):** `{result['confidence']:.1%}`\n\n"
            f"**Policy Switch Point:** Confidence > `{result['switch_p']:.1%}` triggers MOVE\n\n"
            f"**Expected Q-Values:**\n"
            f"  • Move: `{result['expected']['Move']:.2f}`\n"
            f"  • Relocalize: `{result['expected']['Relocalize']:.2f}`\n\n"
            f"*{'Confidence is high enough: robot proceeds to goal.' if action == 'Move' else 'Uncertainty too high: robot requests another reading.'}*"
        )

        return fig, hmm_summary, bayes_summary, astar_summary, mdp_summary
    except Exception as e:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, f"Error: {e}", ha="center", va="center", color="red", fontsize=12)
        ax.axis("off")
        return fig, f"Error: {e}", "", "", ""


def load_preset(name):
    if name == "Test 1: Consistent readings (Move)":
        return "[(4, 3), (4, 3), (4, 3)]", ["east side", "north part"]
    elif name == "Test 2: Contradicting readings (Relocalize)":
        return "[(4, 3), (0, 0), (5, 5)]", ["east side", "north part"]
    elif name == "Test 3: Single noisy reading":
        return "[(4, 4)]", ["east side"]
    return "[(4, 3), (4, 3), (4, 3)]", ["east side", "north part"]


CUSTOM_CSS = """
.container { max-width: 1100px; margin: auto; }
.output-card { border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; background: #f8fafc; }
"""

with gr.Blocks(title="Campus Delivery Robot Demo", css=CUSTOM_CSS) as demo:
    gr.Markdown(
        """
        # 🤖 Campus Delivery Robot: Autonomous Indoor Navigation
        ### AI Fundamentals Final Project · QAIRU (Astana, 2026)
        **Student:** Nurzhas Amangeldi | **Topic:** Multi-Algorithm Decision Making Under Uncertainty
        
        *Reaching a classroom when neither the robot nor 2GIS knows exactly where it is.*
        Integrates **HMM** (Localization) + **Bayesian Inference** (Target Room) + **A\*** (Shortest Route) + **MDP** (Move vs. Relocalize).
        """
    )

    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### 1. Control & Input Configuration")
            preset_dd = gr.Dropdown(
                label="Choose Evaluation Preset",
                choices=[
                    "Test 1: Consistent readings (Move)",
                    "Test 2: Contradicting readings (Relocalize)",
                    "Test 3: Single noisy reading",
                ],
                value="Test 1: Consistent readings (Move)",
            )
            readings_input = gr.Textbox(
                label="Noisy 2GIS Readings (sequence of grid cells)",
                value="[(4, 3), (4, 3), (4, 3)]",
                info="Format: list of (row, col) tuples on 6x6 grid. Walkable cells only.",
            )
            clues_input = gr.CheckboxGroup(
                label="Verbal Clues (Crowdsourced student reports)",
                choices=["east side", "north part"],
                value=["east side", "north part"],
            )
            run_btn = gr.Button("🚀 Run Integrated Navigation Pipeline", variant="primary")

            gr.Markdown("### 2. Map Legend")
            gr.Markdown(
                """
                - **Dark Grey (#)**: Campus Buildings / Walls (impassable)
                - **Blue Cells**: HMM belief distribution heatmap (darker = higher probability)
                - **Red Circle (R)**: Most likely robot position (HMM estimate)
                - **Orange Letters (A, B, C)**: Candidate classroom coordinates (Bayes target is highlighted)
                - **Teal Line**: Optimal collision-free path computed by A*
                """
            )

        with gr.Column(scale=6):
            gr.Markdown("### Campus Grid & Planned Path")
            plot_output = gr.Plot(label="Campus Visualization")

    gr.Markdown("---")
    gr.Markdown("### 3. Algorithm Outputs & Analytical Breakdown")
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 📍 [1] HMM Localisation")
            hmm_box = gr.Markdown()
        with gr.Column():
            gr.Markdown("#### 🎯 [2] Bayesian Room Inference")
            bayes_box = gr.Markdown()
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### 🗺️ [3] A* Path Planning")
            astar_box = gr.Markdown()
        with gr.Column():
            gr.Markdown("#### ⚖️ [4] MDP Decision Making")
            mdp_box = gr.Markdown()

    # Callbacks
    preset_dd.change(
        fn=load_preset,
        inputs=[preset_dd],
        outputs=[readings_input, clues_input],
    )

    run_btn.click(
        fn=solve_robot,
        inputs=[readings_input, clues_input],
        outputs=[plot_output, hmm_box, bayes_box, astar_box, mdp_box],
    )

    demo.load(
        fn=solve_robot,
        inputs=[readings_input, clues_input],
        outputs=[plot_output, hmm_box, bayes_box, astar_box, mdp_box],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
