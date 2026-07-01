"""Visualize steering results for nano: baseline + steered stated + steered revealed."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

VALUE_CLASSES = [
    "Privacy", "Justice", "Respect", "Truthfulness", "Equal Treatment",
    "Protection", "Wisdom", "Care", "Freedom", "Professionalism",
    "Cooperation", "Sustainability", "Learning", "Adaptability",
    "Creativity", "Communication",
]


def rank_stated(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path).set_index("value_class")
    d["rank"] = d["win_rate"].rank(ascending=False, method="min").astype(int)
    return d["rank"].to_dict()


def rank_revealed(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path).set_index("value_class")
    d["rank"] = d["Elo Rating"].rank(ascending=False, method="min").astype(int)
    return d["rank"].to_dict()


def rho(a, b):
    return spearmanr([a[v] for v in VALUE_CLASSES], [b[v] for v in VALUE_CLASSES]).correlation


def render(rows, labels, col_order, title, out_path, rho_target_label="nano's revealed ranking"):
    ordered_vals = [VALUE_CLASSES[i] for i in col_order]
    matrix = np.array([[r[v] for v in VALUE_CLASSES] for r in rows])[:, col_order]

    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.7), max(4, n_rows * 0.5)))
    ax.imshow(matrix, cmap="RdBu", vmin=1, vmax=16, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            v = matrix[i, j]
            color = "white" if v <= 5 or v >= 12 else "black"
            ax.text(j, i, str(int(v)), ha="center", va="center", color=color, fontsize=9)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(ordered_vals, rotation=45, ha="left", fontsize=9)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(-.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    ax.set_title(title, fontsize=11, pad=35)
    plt.figtext(0.5, 0.02,
                f"Columns ordered by nano's baseline revealed ranking. "
                f"ρ = Spearman correlation vs. {rho_target_label}.",
                ha="center", fontsize=8, style="italic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


nano_revealed_base = rank_revealed("elo_rating/gpt-5.4-nano.csv")
assert nano_revealed_base, "baseline revealed ELO missing"
col_order = list(np.argsort([nano_revealed_base[v] for v in VALUE_CLASSES]))

rows_stated, labels_stated = [], []

base_stated = rank_stated("stated_preferences/gpt-5.4-nano_ranking.csv")
if base_stated:
    rows_stated.append(base_stated)
    labels_stated.append(f"baseline stated ({rho(base_stated, nano_revealed_base):+.3f})")

b1 = rank_stated("stated_preferences_steered/revealed_target/gpt-5.4-nano_ranking.csv")
if b1:
    rows_stated.append(b1)
    labels_stated.append(f"steered → revealed ({rho(b1, nano_revealed_base):+.3f})")

b2 = rank_stated("stated_preferences_steered/reversed_revealed_target/gpt-5.4-nano_ranking.csv")
if b2:
    rows_stated.append(b2)
    labels_stated.append(f"steered → reversed rev. ({rho(b2, nano_revealed_base):+.3f})")

rows_stated.append(nano_revealed_base)
labels_stated.append("nano revealed (target)")

render(rows_stated, labels_stated, col_order,
       "Steering nano's stated preferences (B-direction)",
       "heatmap_steering_stated.png")

rows_revealed, labels_revealed = [], []
base_stated_copy = base_stated
if base_stated_copy:
    rows_revealed.append(base_stated_copy)
    labels_revealed.append("nano stated (target for A1)")

a1 = rank_revealed("elo_rating_steered/stated_target/gpt-5.4-nano.csv")
a2 = rank_revealed("elo_rating_steered/reversed_stated_target/gpt-5.4-nano.csv")

rows_revealed.append(nano_revealed_base)
labels_revealed.append(f"baseline revealed ({rho(nano_revealed_base, base_stated):+.3f})")

if a1:
    rows_revealed.append(a1)
    labels_revealed.append(f"steered → stated ({rho(a1, base_stated):+.3f})")
if a2:
    rows_revealed.append(a2)
    labels_revealed.append(f"steered → reversed stated ({rho(a2, base_stated):+.3f})")

if a1 or a2:
    render(rows_revealed, labels_revealed, col_order,
           "Steering nano's revealed preferences (A-direction)",
           "heatmap_steering_revealed.png",
           rho_target_label="nano's stated ranking")
else:
    print("skipped heatmap_steering_revealed.png — A1/A2 not yet complete")
