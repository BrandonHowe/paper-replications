import glob
import os

import pandas as pd
from scipy.stats import spearmanr


VALUE_CLASSES = [
    "Privacy", "Justice", "Respect", "Truthfulness", "Equal Treatment",
    "Protection", "Wisdom", "Care", "Freedom", "Professionalism",
    "Cooperation", "Sustainability", "Learning", "Adaptability",
    "Creativity", "Communication",
]

SHORT_NAMES = {
    "claude-haiku-4-5": "haiku-4.5",
    "claude-opus-4-7":  "opus-4.7",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-5.4-nano": "nano",
    "gpt-5.4": "gpt-5.4",
    "Sonnet 3.7 (paper)": "sonnet-3.7*",
    "GPT-4o (paper)": "gpt-4o*",
}

# Paper data transcribed from Figure 4 (arXiv:2505.14633).
PAPER_RANKINGS = {
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
    for path in sorted(glob.glob("stated_preferences/*_ranking.csv")):
        name = os.path.basename(path).replace("_ranking.csv", "")
        models.setdefault(name, {})["stated"] = load_stated(path)
    for path in sorted(glob.glob("elo_rating/*.csv")):
        name = os.path.basename(path).replace(".csv", "")
        models.setdefault(name, {})["revealed"] = load_revealed(path)
    for name, ranks in PAPER_RANKINGS.items():
        models[name] = ranks
    return models


def spearman(a, b, keys=VALUE_CLASSES):
    if a is None or b is None:
        return None
    return spearmanr([a[v] for v in keys], [b[v] for v in keys]).correlation


def fmt_rank(r):
    return "-" if r is None else str(r)


def short(name):
    return SHORT_NAMES.get(name, name)


def print_table(title, models, kind):
    print(f"\n{title}")
    model_names = [m for m in models if kind in models[m]]
    header_labels = [short(m) for m in model_names]
    v_w = max(len(v) for v in VALUE_CLASSES)
    col_w = max(max(len(h) for h in header_labels), 3)

    def row(first, cells):
        return first.ljust(v_w) + " " + " ".join(c.rjust(col_w) for c in cells)

    print(row("value", header_labels))
    print("-" * v_w + " " + " ".join("-" * col_w for _ in header_labels))
    for v in VALUE_CLASSES:
        cells = [fmt_rank(models[m].get(kind, {}).get(v)) for m in model_names]
        print(row(v, cells))


def print_within_model_consistency(models):
    print("\nSTATED vs REVEALED (within-model Spearman ρ)")
    name_w = max(len(short(n)) for n in models)
    for name, ranks in models.items():
        s = ranks.get("stated")
        r = ranks.get("revealed")
        label = short(name).ljust(name_w)
        if s and r:
            print(f"  {label}  ρ = {spearman(s, r):+.3f}")
        else:
            missing = "stated" if r else "revealed" if s else "both"
            print(f"  {label}  (missing {missing})")


def print_cross_model_matrix(models, kind, title):
    have = [m for m in models if kind in models[m]]
    if len(have) < 2:
        return
    print(f"\n{title}")
    labels = [short(m) for m in have]
    name_w = max(len(l) for l in labels)
    col_w = max(name_w, 6)
    print(" " * name_w + " " + " ".join(l.rjust(col_w) for l in labels))
    for m1, l1 in zip(have, labels):
        cells = []
        for m2 in have:
            if m1 == m2:
                cells.append("--".rjust(col_w))
            else:
                rho = spearman(models[m1][kind], models[m2][kind])
                cells.append(f"{rho:+.3f}".rjust(col_w))
        print(l1.ljust(name_w) + " " + " ".join(cells))


if __name__ == "__main__":
    models = discover_models()
    if not models:
        print("No model data found. Run the generation scripts first.")
    else:
        print_table("STATED PREFERENCES (rank 1 = most prioritized)", models, "stated")
        print_table("REVEALED PREFERENCES (rank 1 = most prioritized)", models, "revealed")
        print_within_model_consistency(models)
        print_cross_model_matrix(models, "stated", "STATED cross-model Spearman ρ")
        print_cross_model_matrix(models, "revealed", "REVEALED cross-model Spearman ρ")
        print("\n* = data from paper (arXiv:2505.14633)")
