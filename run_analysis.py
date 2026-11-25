import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mfa_model import (
    BASE_SEED,
    CONFIG,
    FLOWS,
    FLOW_KEY,
    LABELS,
    MAIN_OUTPUTS,
    MODES,
    N_ITERATIONS,
    SCORES,
    apply_dependence,
    cv_exponential,
    cv_linear,
    interval_width,
    parameter_bounds,
    run_model,
    sample_inputs,
    simulate,
    summary_statistics,
    total_cv,
)

FORMAL_FRACTIONS = ["a_rec", "a_reu", "a_en"]

DEPENDENCE_CASES = [
    ("formal_rho_-0.25", FORMAL_FRACTIONS, -0.25),
    ("formal_rho_-0.45", FORMAL_FRACTIONS, -0.45),
    ("formal_rho_+0.50", FORMAL_FRACTIONS, 0.50),
    ("formal_rho_+0.80", FORMAL_FRACTIONS, 0.80),
    ("collection_informal_rho_+0.50", ["c", "phi"], 0.50),
    ("collection_informal_rho_-0.50", ["c", "phi"], -0.50),
]

SENSITIVITY_INPUTS = {
    "Urban plastics waste": "W",
    "Collection efficiency": "c",
    "Informal share": "phi",
    "Formal recycling fraction": "a_rec",
    "Formal energy fraction": "a_en",
    "Formal reutilisation fraction": "a_reu",
    "Informal recovery fraction": "eta",
    "Reutilisation of informal rejects": "tau",
}
SENSITIVITY_OUTPUTS = {
    "uncollected_rejects": "uncollected_rejects",
    "CPI": "CPI",
    "LDS": "LDS",
}
SENSITIVITY_TITLES = [
    "(a) Uncollected waste and informal rejects",
    "(b) CPI",
    "(c) LDS",
]

CONVERGENCE_OUTPUTS = [
    "Recycled",
    "Reutilised",
    "Energy recovery",
    "Landfills and dumpsites",
    "Uncollected waste and informal rejects",
]
N_SEEDS = 20
SEED_OFFSET = 1000
N_LARGE_RUN = 100000


def save_parameter_tables(out_dir):
    rows = []
    for key in MODES:
        low_exp, high_exp = parameter_bounds(key, "exp")
        low_lin, high_lin = parameter_bounds(key, "lin")
        rows.append(
            {
                "Parameter": LABELS[key],
                "Scores_RCTGO": "".join(map(str, SCORES[key])),
                "Mode": MODES[key],
                "CV_total_exp": round(total_cv(SCORES[key], "exp"), 3),
                "Low_exp": low_exp,
                "High_exp": high_exp,
                "CV_total_lin": round(total_cv(SCORES[key], "lin"), 3),
                "Low_lin": low_lin,
                "High_lin": high_lin,
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "parameter_ranges.csv"), index=False)

    pd.DataFrame(
        {
            "Score": [1, 2, 3, 4],
            "Exp_reliability_%": [cv_exponential(s, True) for s in range(1, 5)],
            "Exp_other_indicators_%": [cv_exponential(s) for s in range(1, 5)],
            "Lin_reliability_%": [cv_linear(s, True) for s in range(1, 5)],
            "Lin_other_indicators_%": [cv_linear(s) for s in range(1, 5)],
        }
    ).to_csv(os.path.join(out_dir, "cv_by_score.csv"), index=False)

    bounds_sum = [
        {"Run": kind, "Sum_of_upper_bounds": sum(parameter_bounds(k, kind)[1] for k in FORMAL_FRACTIONS)}
        for kind in ("exp", "lin")
    ]
    pd.DataFrame(bounds_sum).to_csv(os.path.join(out_dir, "formal_fraction_bounds_sum.csv"), index=False)


def flow_table(out, run):
    rows = []
    for name, key in FLOWS:
        rows.append({"Run": run, "Flow": name, **summary_statistics(out[key])})
    return pd.DataFrame(rows)


def save_uncertainty_results(runs, out_dir):
    table = pd.concat([flow_table(runs[r][1], r) for r in ("exp", "lin")], ignore_index=True)
    table.to_csv(os.path.join(out_dir, "uncertainty_results.csv"), index=False)

    rows = []
    for name in MAIN_OUTPUTS:
        row = {"Output": name}
        for run in ("exp", "lin"):
            values = runs[run][1][FLOW_KEY[name]]
            row[f"Width_{run}"] = interval_width(values)
            row[f"Width_relative_to_mean_{run}_%"] = 100 * interval_width(values) / values.mean()
        row["Width_ratio_lin_vs_exp"] = row["Width_lin"] / row["Width_exp"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "interval_widths.csv"), index=False)

    rows = []
    for key in MODES:
        row = {"Parameter": LABELS[key], "Mode": MODES[key]}
        for run in ("exp", "lin"):
            row[f"Sample_mean_{run}"] = runs[run][0][key].mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "input_mean_vs_mode.csv"), index=False)


def sensitivity_table(inputs, out):
    rows = []
    for label, key in SENSITIVITY_INPUTS.items():
        row = {"Parameter": label}
        for name, output_key in SENSITIVITY_OUTPUTS.items():
            row[f"rho_{name}"] = spearmanr(inputs[key], out[output_key])[0]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sensitivity(table, path):
    order = (
        table.reindex(table["rho_uncollected_rejects"].abs().sort_values(ascending=False).index)[
            "Parameter"
        ].tolist()
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    columns = ["rho_uncollected_rejects", "rho_CPI", "rho_LDS"]
    indexed = table.set_index("Parameter")
    for ax, column, title in zip(axes, columns, SENSITIVITY_TITLES):
        values = indexed.loc[order, column]
        ax.barh(values.index, values.values, color="cadetblue")
        ax.axvline(0, ls="--", c="k", lw=0.8)
        ax.set_title(title, loc="left")
        ax.set_xlabel(r"Spearman $\rho$")
        ax.invert_yaxis()
        if column == "rho_LDS":
            ax.set_xlim(right=0.05)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_sensitivity(runs, out_dir):
    for run in ("exp", "lin"):
        table = sensitivity_table(*runs[run])
        table.to_csv(os.path.join(out_dir, f"sensitivity_{run}.csv"), index=False)
        if run == "exp":
            plot_sensitivity(table, os.path.join(out_dir, "sensitivity.png"))


def run_dependence(out_dir):
    base_inputs = sample_inputs(CONFIG["exp"], N_ITERATIONS, np.random.default_rng(BASE_SEED))
    base_out = run_model(base_inputs)
    rows, achieved = [], []
    for case, keys, rho in DEPENDENCE_CASES:
        inputs = apply_dependence(base_inputs, keys, rho, np.random.default_rng(BASE_SEED + 1))
        out = run_model(inputs)
        achieved.append(
            {
                "Case": case,
                "Target_rho": rho,
                "Achieved_rho_first_pair": round(spearmanr(inputs[keys[0]], inputs[keys[1]])[0], 3),
            }
        )
        for name in MAIN_OUTPUTS:
            before, after = base_out[FLOW_KEY[name]], out[FLOW_KEY[name]]
            rows.append(
                {
                    "Case": case,
                    "Output": name,
                    "Mean": after.mean(),
                    "CI95_low": np.percentile(after, 2.5),
                    "CI95_high": np.percentile(after, 97.5),
                    "Mean_change_%": 100 * (after.mean() / before.mean() - 1),
                    "Width_change_%": 100 * (interval_width(after) / interval_width(before) - 1),
                }
            )
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "dependence_cases.csv"), index=False)
    pd.DataFrame(achieved).to_csv(os.path.join(out_dir, "dependence_achieved_rho.csv"), index=False)


def run_convergence(runs, out_dir):
    base_out = runs["exp"][1]
    checkpoints = np.unique(np.round(np.logspace(2, math.log10(N_ITERATIONS), 40)).astype(int))
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name in CONVERGENCE_OUTPUTS:
        x = base_out[FLOW_KEY[name]]
        cumulative, cumulative_sq = np.cumsum(x), np.cumsum(x * x)
        n = checkpoints
        mean = cumulative[n - 1] / n
        sd = np.sqrt(np.maximum((cumulative_sq[n - 1] - cumulative[n - 1] ** 2 / n) / (n - 1), 0))
        half_width = 1.96 * sd / np.sqrt(n)
        for count, m, h in zip(n, mean, half_width):
            rows.append(
                {
                    "Output": name,
                    "Iterations": count,
                    "Running_mean": m,
                    "Half_width_95": h,
                    "Half_width_%_of_mean": 100 * h / m,
                }
            )
        axes[0].plot(n, mean / mean[-1], label=name)
        axes[1].plot(n, 100 * half_width / mean, label=name)
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Running mean / mean at 10,000 iterations")
    axes[0].legend()
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Half-width of 95% CI of the mean (% of mean)")
    axes[0].set_title("(a)", loc="left")
    axes[1].set_title("(b)", loc="left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence_running_mean.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "convergence_running.csv"), index=False)

    seed_rows = []
    for s in range(1, N_SEEDS + 1):
        _, out = simulate(CONFIG["exp"], seed=SEED_OFFSET + s)
        for name in MAIN_OUTPUTS:
            seed_rows.append({"Seed": SEED_OFFSET + s, "Output": name, **summary_statistics(out[FLOW_KEY[name]])})
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(os.path.join(out_dir, "convergence_seeds_all.csv"), index=False)
    grouped = seeds.groupby("Output")
    summary = pd.DataFrame(
        {
            "Mean_of_means": grouped["Mean"].mean(),
            "SD_of_mean_%_of_mean": 100 * grouped["Mean"].std() / grouped["Mean"].mean(),
            "Range_P2.5_%": 100 * (grouped["CI95_low"].max() - grouped["CI95_low"].min()) / grouped["CI95_low"].mean(),
            "Range_P97.5_%": 100 * (grouped["CI95_high"].max() - grouped["CI95_high"].min()) / grouped["CI95_high"].mean(),
        }
    ).loc[MAIN_OUTPUTS]
    summary.to_csv(os.path.join(out_dir, "convergence_seeds_summary.csv"))

    _, large = simulate(CONFIG["exp"], n=N_LARGE_RUN, seed=BASE_SEED)
    rows = []
    for name in MAIN_OUTPUTS:
        small_values, large_values = base_out[FLOW_KEY[name]], large[FLOW_KEY[name]]
        rows.append(
            {
                "Output": name,
                "Mean_diff_%": 100 * (large_values.mean() / small_values.mean() - 1),
                "P2.5_diff_%": 100 * (np.percentile(large_values, 2.5) / np.percentile(small_values, 2.5) - 1),
                "P97.5_diff_%": 100 * (np.percentile(large_values, 97.5) / np.percentile(small_values, 97.5) - 1),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "convergence_100k_vs_10k.csv"), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--skip-convergence", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = {kind: simulate(CONFIG[kind]) for kind in ("exp", "lin")}
    save_parameter_tables(args.out)
    save_uncertainty_results(runs, args.out)
    run_sensitivity(runs, args.out)
    run_dependence(args.out)
    if not args.skip_convergence:
        run_convergence(runs, args.out)


if __name__ == "__main__":
    main()
