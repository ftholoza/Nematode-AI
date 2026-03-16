import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from brain import Brain


# =====================================
# Config
# =====================================
OUTPUT_DIR = "visualize"
OUTPUT_MP4 = os.path.join(OUTPUT_DIR, "brain_activity.mp4")

N_FRAMES = 220
FRAME_INTERVAL_MS = 120

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================
# Brain
# =====================================
brain = Brain(alpha=0.2)


# =====================================
# Build graph from brain
# =====================================
G = nx.DiGraph()

for neuron_id, neuron_name in brain.id_to_name.items():
    G.add_node(
        neuron_id,
        name=neuron_name,
        neuron_type=brain.id_to_type[neuron_id],
    )

for post in range(brain.n):
    for pre in range(brain.n):
        w = brain.W[post, pre]
        if abs(w) > 1e-12:
            G.add_edge(pre, post, weight=w)


# =====================================
# Layout by neuron type
# =====================================
def build_layout(brain: Brain):
    sensory = [i for i in range(brain.n) if brain.id_to_type[i] == "sensory"]
    inter = [i for i in range(brain.n) if brain.id_to_type[i] == "inter"]
    motor = [i for i in range(brain.n) if brain.id_to_type[i] == "motor"]

    pos = {}

    def place_vertical(ids, x):
        if len(ids) == 1:
            ys = [0.0]
        else:
            ys = np.linspace(1.5, -1.5, len(ids))
        for i, nid in enumerate(ids):
            pos[nid] = (x, ys[i])

    place_vertical(sensory, -2.5)
    place_vertical(inter, 0.0)
    place_vertical(motor, 2.5)

    return pos


pos = build_layout(brain)


# =====================================
# Synthetic sensory stimulation
# =====================================
def make_sensory_input(frame: int):
    """
    Simple demo input pattern for the 14-neuron brain.

    Phase 1:
        target more on the left and far
    Phase 2:
        target more centered and closer
    Phase 3:
        target more on the right and near
    """
    sensory_values = {
        "S_left_far": 0.0,
        "S_left_near": 0.0,
        "S_right_far": 0.0,
        "S_right_near": 0.0,
    }

    if frame < 70:
        sensory_values["S_left_far"] = 0.85
        sensory_values["S_left_near"] = 0.10
        sensory_values["S_right_far"] = 0.25
        sensory_values["S_right_near"] = 0.00

    elif frame < 140:
        sensory_values["S_left_far"] = 0.55
        sensory_values["S_left_near"] = 0.35
        sensory_values["S_right_far"] = 0.50
        sensory_values["S_right_near"] = 0.30

    else:
        sensory_values["S_left_far"] = 0.20
        sensory_values["S_left_near"] = 0.15
        sensory_values["S_right_far"] = 0.65
        sensory_values["S_right_near"] = 0.70

    return sensory_values


# =====================================
# Figure
# =====================================
fig, (ax_graph, ax_info) = plt.subplots(
    1,
    2,
    figsize=(14, 8),
    gridspec_kw={"width_ratios": [3.3, 1.5]},
)

ax_info.axis("off")


# =====================================
# Animation
# =====================================
def update(frame: int):
    sensory_values = make_sensory_input(frame)
    sensory = brain.build_sensory_vector(sensory_values)

    x, total_input = brain.step(sensory)

    ax_graph.clear()
    ax_info.clear()
    ax_info.axis("off")

    # Node colors = current activity
    node_colors = [x[i] for i in range(brain.n)]

    # Labels
    node_labels = {}
    for node in G.nodes:
        name = G.nodes[node]["name"]
        neuron_type = G.nodes[node]["neuron_type"]
        activity = x[node]
        node_labels[node] = f"{name}\n{neuron_type}\nx={activity:.2f}"

    # Edge colors / labels
    edge_colors = []
    edge_widths = []
    edge_labels = {}

    for u, v, data in G.edges(data=True):
        w = data["weight"]
        edge_colors.append("green" if w > 0 else "red")
        edge_widths.append(1.5 + 1.2 * abs(w))
        edge_labels[(u, v)] = f"{w:.1f}"

    # Draw graph
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        cmap="plasma",
        vmin=-1,
        vmax=1,
        node_size=2600,
        ax=ax_graph,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_size=8,
        ax=ax_graph,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=22,
        connectionstyle="arc3,rad=0.08",
        ax=ax_graph,
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        ax=ax_graph,
    )

    ax_graph.set_title(f"14-neuron circuit activity — t={frame}")
    ax_graph.axis("off")

    # Info panel
    info_text = f"""
Sensory input
L_far   = {sensory_values["S_left_far"]:.3f}
L_near  = {sensory_values["S_left_near"]:.3f}
R_far   = {sensory_values["S_right_far"]:.3f}
R_near  = {sensory_values["S_right_near"]:.3f}

Interneurons
I_left  = {brain.get_activity_by_name("I_left"):.3f}
I_right = {brain.get_activity_by_name("I_right"):.3f}
I_fwd   = {brain.get_activity_by_name("I_forward"):.3f}
I_srch  = {brain.get_activity_by_name("I_search"):.3f}
I_memL  = {brain.get_activity_by_name("I_memory_left"):.3f}
I_memR  = {brain.get_activity_by_name("I_memory_right"):.3f}

Motors
M_left  = {brain.get_activity_by_name("M_left"):.3f}
M_right = {brain.get_activity_by_name("M_right"):.3f}
M_fwd   = {brain.get_activity_by_name("M_forward"):.3f}
M_brake = {brain.get_activity_by_name("M_brake"):.3f}
"""

    ax_info.text(
        0.0,
        0.98,
        info_text,
        va="top",
        ha="left",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.92),
    )


ani = FuncAnimation(
    fig,
    update,
    frames=N_FRAMES,
    interval=FRAME_INTERVAL_MS,
    repeat=False,
)

ani.save(
    OUTPUT_MP4,
    writer="ffmpeg",
    fps=30,
    bitrate=1800,
)

print(f"Saved animation to {OUTPUT_MP4}")