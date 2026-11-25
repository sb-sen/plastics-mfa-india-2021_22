import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

from mfa_model import CONFIG, FLOW_KEY, MAIN_OUTPUTS, summary_statistics
from scenarios import R_PACK, SCENARIOS, simulate_scenarios

VARIABLES = ["Urban plastics waste"] + MAIN_OUTPUTS
FLOW_VARIABLES = [v for v in VARIABLES if v not in ("CPI", "LDS")]
PLOT_SCENARIOS = [
    "Baseline",
    "A1",
    "A2",
    "B (zeta=0.83)",
    "B (zeta=1)",
    "C (d=0.5, chi=0.5)",
    "C (d=1, chi=0.5)",
    "D1",
    "D2",
]
PATHWAYS = [
    ("Recycled", "recycled", "#00A087"),
    ("Reutilised", "reutilised", "#91D1C2"),
    ("Energy recovery", "energy", "#4DBBD5"),
    ("Landfills and dumpsites", "landfilled", "#8491B4"),
    ("Uncollected waste and informal rejects", "uncollected_rejects", "#E64B35"),
]


def build_table(results):
    base = results["Baseline"]
    rows = []
    for scenario, out in results.items():
        for name in VARIABLES:
            key = FLOW_KEY[name]
            values = out[key]
            stats = summary_statistics(values)
            diff = summary_statistics(values - base[key])
            rows.append(
                {
                    "Scenario": scenario,
                    "Variable": name,
                    "Mean": stats["Mean"],
                    "Median": stats["Median"],
                    "CI95_low": stats["CI95_low"],
                    "CI95_high": stats["CI95_high"],
                    "Diff_mean": diff["Mean"],
                    "Diff_CI95_low": diff["CI95_low"],
                    "Diff_CI95_high": diff["CI95_high"],
                }
            )
    return pd.DataFrame(rows)


SHOWN = {
    "A1": ("A1", "#8491B4"),
    "A2": ("A2", "#F39B7F"),
    "B (zeta=0.83)": ("B", "#3C5488"),
    "C (d=0.5, chi=0.5)": ("C", "#4DBBD5"),
    "D1": ("D1", "#00A087"),
    "D2": ("D2", "#E64B35"),
}
BASE_COLOR = "#4d4d4d"
LABEL_OFFSET = {"A2": (-6, 12), "B": (-18, 8), "C": (8, -14), "D1": (8, -14), "D2": (8, 6)}


def style_axis(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def confidence_ellipse(ax, x, y, color):
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * np.sqrt(5.991 * vals)
    ax.add_patch(Ellipse((x.mean(), y.mean()), width, height, angle=angle,
                         facecolor=color, edgecolor=color, alpha=0.16, linewidth=0.8))


def plot_overview(results, path):
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})
    base = results["Baseline"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.6), gridspec_kw={"width_ratios": [1, 1.05]})

    xs = np.linspace(0.0, 1.0, 50)
    ax1.fill_between(xs, 1 - xs, 1.2, color="#f2f2f2", zorder=0)
    ax1.plot(xs, 1 - xs, color="#a0a0a0", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    ax1.text(0.56, 0.90, "CPI + LDS > 1\nnot possible", fontsize=8.5, color="#8a8a8a", ha="left", va="top")
    bx, by = base["LDS"].mean(), base["CPI"].mean()
    for key, (label, color) in SHOWN.items():
        if label == "A1":
            continue
        lds, cpi = results[key]["LDS"], results[key]["CPI"]
        x, y = lds.mean(), cpi.mean()
        ax1.plot(np.percentile(lds, [2.5, 97.5]), [y, y], color=color, linewidth=1.0, alpha=0.55, zorder=2)
        ax1.plot([x, x], np.percentile(cpi, [2.5, 97.5]), color=color, linewidth=1.0, alpha=0.55, zorder=2)
        ax1.annotate("", xy=(x, y), xytext=(bx, by),
                     arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, shrinkA=5, shrinkB=3, mutation_scale=12), zorder=3)
        ax1.scatter([x], [y], s=50, color=color, edgecolor="white", linewidth=0.8, zorder=4)
        ax1.annotate(label, (x, y), textcoords="offset points", xytext=LABEL_OFFSET.get(label, (7, 5)),
                     fontsize=10.5, color=color, fontweight="bold")
    ax1.plot(np.percentile(base["LDS"], [2.5, 97.5]), [by, by], color=BASE_COLOR, linewidth=1.0, alpha=0.55, zorder=2)
    ax1.plot([bx, bx], np.percentile(base["CPI"], [2.5, 97.5]), color=BASE_COLOR, linewidth=1.0, alpha=0.55, zorder=2)
    ax1.scatter([bx], [by], s=80, color=BASE_COLOR, edgecolor="white", linewidth=0.8, zorder=5)
    ax1.annotate("Baseline = A1", (bx, by), textcoords="offset points", xytext=(8, -16), fontsize=10.5, color=BASE_COLOR, fontweight="bold")
    ax1.set_xlim(0.68, -0.04)
    ax1.set_ylim(0.15, 0.98)
    ax1.set_xlabel("Linear Disposal Share (LDS)")
    ax1.set_ylabel("Circularity Performance Indicator (CPI)")
    ax1.set_title("(a)", loc="left", fontsize=12, fontweight="bold")
    style_axis(ax1)

    names = [label for label, _ in SHOWN.values()]
    x = np.arange(len(SHOWN))
    pos_bottom = np.zeros(len(SHOWN))
    neg_bottom = np.zeros(len(SHOWN))
    for label, key, color in PATHWAYS:
        deltas = np.array([(results[k][key] - base[key]).mean() / 1e6 for k in SHOWN])
        pos = np.where(deltas > 0, deltas, 0.0)
        neg = np.where(deltas < 0, deltas, 0.0)
        ax2.bar(x, pos, bottom=pos_bottom, color=color, width=0.62, label=label, edgecolor="white", linewidth=0.5)
        ax2.bar(x, neg, bottom=neg_bottom, color=color, width=0.62, edgecolor="white", linewidth=0.5)
        pos_bottom += pos
        neg_bottom += neg
    net = [(sum(results[k][key] for _, key, _ in PATHWAYS) - sum(base[key] for _, key, _ in PATHWAYS)).mean() / 1e6 for k in SHOWN]
    ax2.scatter(x, net, color="black", s=26, zorder=5, label="Net change")
    ax2.axhline(0, color="#404040", linewidth=0.8)
    ax2.set_ylim(neg_bottom.min() - 0.12, pos_bottom.max() + 0.12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Change in annual flow relative to baseline (Mt per year)")
    ax2.set_xlabel("Scenario")
    ax2.set_title("(b)", loc="left", fontsize=12, fontweight="bold")
    ax2.yaxis.grid(True, color="#ebebeb", linewidth=0.6)
    ax2.set_axisbelow(True)
    style_axis(ax2)
    handles, labels = ax2.get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"Packaging recycling target R_pack = {R_PACK:.3f}")
    for kind in ("exp", "lin"):
        results = simulate_scenarios(CONFIG[kind])
        table = build_table(results)
        table.to_csv(os.path.join(args.out, f"scenario_results_{kind}.csv"), index=False)
        if kind == "exp":
            plot_overview(results, os.path.join(args.out, "scenarios_overview.png"))
        print(f"\nScenario results ({kind})")
        for scenario in results:
            sub = table[table["Scenario"] == scenario].set_index("Variable")
            parts = []
            for name in ("Recycled", "Landfills and dumpsites", "Uncollected waste and informal rejects"):
                parts.append(f"{name.split(' ')[0]} {sub.loc[name, 'Mean'] / 1e6:.2f} Mt")
            parts.append(f"CPI {sub.loc['CPI', 'Mean']:.3f} [{sub.loc['CPI', 'CI95_low']:.3f}, {sub.loc['CPI', 'CI95_high']:.3f}]")
            parts.append(f"LDS {sub.loc['LDS', 'Mean']:.3f} [{sub.loc['LDS', 'CI95_low']:.3f}, {sub.loc['LDS', 'CI95_high']:.3f}]")
            print(f"{scenario:<22} " + "; ".join(parts))


if __name__ == "__main__":
    main()
