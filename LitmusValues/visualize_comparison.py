import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VALUE_CLASSES = [
    "Privacy", "Justice", "Respect", "Truthfulness", "Equal Treatment",
    "Protection", "Wisdom", "Care", "Freedom", "Professionalism",
    "Cooperation", "Sustainability", "Learning", "Adaptability",
    "Creativity", "Communication",
]

SHORT_NAMES = {
    "claude-haiku-4-5": "Haiku 4.5",
    "claude-opus-4-7":  "Opus 4.7",
    "gpt-4o-2024-08-06": "GPT-4o",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "gpt-5.4": "GPT-5.4",
    "Sonnet 3.7 (paper)": "Sonnet 3.7 (ref.)",
    "GPT-4o (paper)": "GPT-4o (ref.)",
}

DISPLAY_ORDER = [
    "Sonnet 3.7 (paper)",
    "GPT-4o (paper)",
    "claude-haiku-4-5",
    "claude-opus-4-7",
    "gpt-5.4-nano",
    "gpt-5.4",
]

REVEALED_ORDER = [
    "Sonnet 3.7 (paper)",
    "GPT-4o (paper)",
    "claude-haiku-4-5",
    "claude-opus-4-7",
    "gpt-5.4-nano",
    "gpt-5.4",
]

OUR_MODELS = [m for m in DISPLAY_ORDER if "(paper)" not in m]
PAPER_MODELS = [m for m in DISPLAY_ORDER if "(paper)" in m]

PAPER_RANKINGS = {
    "GPT-4o (paper)": {
        "stated": {
            "Privacy": 14, "Justice": 9, "Respect": 11, "Truthfulness": 1,
            "Equal Treatment": 10, "Protection": 16, "Wisdom": 2, "Care": 5,
            "Freedom": 8, "Professionalism": 15, "Cooperation": 4,
            "Sustainability": 7, "Learning": 6, "Adaptability": 3,
            "Creativity": 12, "Communication": 13,
        },
        "revealed": {
            "Privacy": 1, "Justice": 3, "Respect": 2, "Truthfulness": 4,
            "Equal Treatment": 5, "Protection": 6, "Wisdom": 8, "Care": 7,
            "Freedom": 11, "Professionalism": 9, "Cooperation": 10,
            "Sustainability": 12, "Learning": 14, "Adaptability": 13,
            "Creativity": 16, "Communication": 15,
        },
    },
    "Sonnet 3.7 (paper)": {
        "stated": {
            "Privacy": 14, "Justice": 11, "Respect": 8, "Truthfulness": 4,
            "Equal Treatment": 12, "Protection": 16, "Wisdom": 1, "Care": 6,
            "Freedom": 5, "Professionalism": 15, "Cooperation": 10,
            "Sustainability": 13, "Learning": 2, "Adaptability": 3,
            "Creativity": 9, "Communication": 7,
        },
        "revealed": {
            "Privacy": 1, "Justice": 4, "Respect": 2, "Truthfulness": 3,
            "Equal Treatment": 5, "Protection": 8, "Wisdom": 9, "Care": 11,
            "Freedom": 6, "Professionalism": 7, "Cooperation": 10,
            "Sustainability": 12, "Learning": 14, "Adaptability": 15,
            "Creativity": 16, "Communication": 13,
        },
    },
}


def load_stated(path):
    df = pd.read_csv(path).set_index("value_class")
    df["rank"] = df["win_rate"].rank(ascending=False, method="min").astype(int)
    return df["rank"].to_dict()


def load_revealed(path):
    df = pd.read_csv(path).set_index("value_class")
    df["rank"] = df["Elo Rating"].rank(ascending=False, method="min").astype(int)
    return df["rank"].to_dict()


def discover_models():
    models = {}
    for path in glob.glob("stated_preferences/*_ranking.csv"):
        name = os.path.basename(path).replace("_ranking.csv", "")
        models.setdefault(name, {})["stated"] = load_stated(path)
    for path in glob.glob("elo_rating/*.csv"):
        name = os.path.basename(path).replace(".csv", "")
        models.setdefault(name, {})["revealed"] = load_revealed(path)
    for name, ranks in PAPER_RANKINGS.items():
        models[name] = ranks
    return models


def build_matrix(models, order):
    rows = []
    labels = []
    for m in order:
        if m not in models:
            continue
        short = SHORT_NAMES.get(m, m)
        if "stated" in models[m]:
            rows.append([models[m]["stated"][v] for v in VALUE_CLASSES])
            labels.append(f"(Stated) {short}")
        if "revealed" in models[m]:
            rows.append([models[m]["revealed"][v] for v in VALUE_CLASSES])
            labels.append(f"(Revealed) {short}")
    return np.array(rows), labels


def order_columns_by_revealed(models, order):
    revealed_models = [m for m in order
                       if m in models and "revealed" in models[m]]
    if not revealed_models:
        return list(range(len(VALUE_CLASSES)))
    avg = []
    for v in VALUE_CLASSES:
        ranks = [models[m]["revealed"][v] for m in revealed_models]
        avg.append(np.mean(ranks))
    return list(np.argsort(avg))


def render(matrix, row_labels, col_order, out_path, title=None):
    ordered_values = [VALUE_CLASSES[i] for i in col_order]
    ordered_matrix = matrix[:, col_order]

    n_rows, n_cols = ordered_matrix.shape
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.7), max(4, n_rows * 0.45)))

    im = ax.imshow(ordered_matrix, cmap="RdBu", vmin=1, vmax=16, aspect="auto")

    for i in range(n_rows):
        for j in range(n_cols):
            val = ordered_matrix[i, j]
            color = "white" if val <= 5 or val >= 12 else "black"
            ax.text(j, i, str(int(val)), ha="center", va="center",
                    color=color, fontsize=9)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(ordered_values, rotation=45, ha="left", fontsize=9)
    ax.xaxis.tick_top()

    ax.set_xticks(np.arange(-.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    if title:
        ax.set_title(title, fontsize=11, pad=35)
    plt.figtext(0.5, 0.02,
                "Rank 1 = most prioritized, 16 = least. Columns ordered by average revealed rank.",
                ha="center", fontsize=8, style="italic")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


ALL_MODELS = DISPLAY_ORDER


def consensus_revealed(models):
    """Average rank across all models with revealed data, re-ranked 1-16."""
    have = [m for m in ALL_MODELS if m in models and "revealed" in models[m]]
    means = {v: np.mean([models[m]["revealed"][v] for m in have]) for v in VALUE_CLASSES}
    sorted_values = sorted(VALUE_CLASSES, key=lambda v: means[v])
    return {v: i + 1 for i, v in enumerate(sorted_values)}


def build_stated_matrix_with_consensus(models, consensus):
    from scipy.stats import spearmanr
    rows, labels = [], []
    for m in ALL_MODELS:
        if m not in models or "stated" not in models[m]:
            continue
        stated = models[m]["stated"]
        rev = models[m].get("revealed") or models.get("claude-haiku-4-5", {}).get("revealed")
        if rev:
            rho = spearmanr([stated[v] for v in VALUE_CLASSES],
                            [rev[v] for v in VALUE_CLASSES]).correlation
            label = f"{SHORT_NAMES.get(m, m)} ({rho:+.3f})"
        else:
            label = SHORT_NAMES.get(m, m)
        rows.append([stated[v] for v in VALUE_CLASSES])
        labels.append(label)
    rows.append([consensus[v] for v in VALUE_CLASSES])
    labels.append("Revealed consensus")
    return np.array(rows), labels


def build_revealed_matrix(models, order=None):
    if order is None:
        order = ALL_MODELS
    rows, labels = [], []
    for m in order:
        if m in models and "revealed" in models[m]:
            rows.append([models[m]["revealed"][v] for v in VALUE_CLASSES])
            labels.append(SHORT_NAMES.get(m, m))
    return np.array(rows), labels


def column_order_from_consensus(consensus):
    return list(np.argsort([consensus[v] for v in VALUE_CLASSES]))


if __name__ == "__main__":
    models = discover_models()
    consensus = consensus_revealed(models)
    col_order = column_order_from_consensus(consensus)

    matrix_s, labels_s = build_stated_matrix_with_consensus(models, consensus)
    if matrix_s.size:
        render(matrix_s, labels_s, col_order, "heatmap_stated.png",
               title="Stated preferences (all models) vs. consensus revealed ranking")

    matrix_r, labels_r = build_revealed_matrix(models, REVEALED_ORDER)
    if matrix_r.size:
        render(matrix_r, labels_r, col_order, "heatmap_revealed.png",
               title="Revealed preferences (all models)")
