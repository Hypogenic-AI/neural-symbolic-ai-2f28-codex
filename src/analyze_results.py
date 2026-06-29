"""Analyze AddMNIST experiment outputs and create figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


METHOD_ORDER = ["direct_sum", "symbolic_sum", "anchor_symbolic", "anchor_posterior"]
METHOD_LABELS = {
    "direct_sum": "Direct CNN",
    "symbolic_sum": "Symbolic",
    "anchor_symbolic": "Anchor symbolic",
    "anchor_posterior": "Anchor posterior",
}


def cohen_dz(diff: np.ndarray) -> float:
    sd = np.std(diff, ddof=1)
    if sd == 0:
        return float("inf") if np.mean(diff) != 0 else 0.0
    return float(np.mean(diff) / sd)


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot[i] = sample.mean()
    return tuple(np.quantile(boot, [0.025, 0.975]).tolist())


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ood = metrics[metrics["split"] == "ood_high_high"].copy()
    comparisons = [
        ("anchor_posterior", "direct_sum"),
        ("anchor_posterior", "symbolic_sum"),
        ("anchor_posterior", "anchor_symbolic"),
        ("anchor_symbolic", "direct_sum"),
        ("symbolic_sum", "direct_sum"),
    ]
    for budget in sorted(ood["budget"].unique()):
        pivot = ood[ood["budget"] == budget].pivot_table(
            index="seed", columns="method", values="accuracy", aggfunc="first"
        )
        for method_a, method_b in comparisons:
            if method_a not in pivot.columns or method_b not in pivot.columns:
                continue
            paired = pivot[[method_a, method_b]].dropna()
            diff = (paired[method_a] - paired[method_b]).to_numpy()
            if len(diff) >= 2:
                test = stats.ttest_rel(paired[method_a], paired[method_b])
                p_value = float(test.pvalue)
                statistic = float(test.statistic)
            else:
                p_value = np.nan
                statistic = np.nan
            rows.append(
                {
                    "budget": budget,
                    "metric": "ood_high_high_accuracy",
                    "method_a": method_a,
                    "method_b": method_b,
                    "mean_a": float(paired[method_a].mean()),
                    "mean_b": float(paired[method_b].mean()),
                    "mean_diff": float(diff.mean()) if len(diff) else np.nan,
                    "cohen_dz": cohen_dz(diff) if len(diff) >= 2 else np.nan,
                    "t_statistic": statistic,
                    "p_value": p_value,
                    "n_seeds": int(len(diff)),
                }
            )
    return pd.DataFrame(rows)


def prediction_bootstrap(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260629)
    rows = []
    for keys, frame in predictions.groupby(["method", "budget", "split"], sort=True):
        method, budget, split = keys
        values = frame["correct"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_ci(values, rng)
        rows.append(
            {
                "method": method,
                "budget": int(budget),
                "split": split,
                "accuracy": float(values.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_predictions": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def ood_sum_region_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    ood = predictions[predictions["split"] == "ood_high_high"].copy()
    ood["sum_region"] = np.where(
        ood["true_sum"] >= 14,
        "unseen_train_label_14_18",
        "seen_sum_10_13",
    )
    return (
        ood.groupby(["method", "budget", "sum_region"], as_index=False)
        .agg(
            accuracy=("correct", "mean"),
            n=("correct", "size"),
            mean_confidence=("confidence", "mean"),
        )
        .sort_values(["budget", "sum_region", "method"])
    )


def add_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["method_label"] = out["method"].map(METHOD_LABELS)
    return out


def plot_accuracy(metrics: pd.DataFrame, figures_dir: Path) -> None:
    pair = metrics[metrics["split"].isin(["iid_restricted", "ood_high_high"])].copy()
    pair = add_method_labels(pair)
    pair["split_label"] = pair["split"].map(
        {"iid_restricted": "IID restricted", "ood_high_high": "OOD high+high"}
    )
    sns.set_theme(style="whitegrid", context="talk")
    g = sns.catplot(
        data=pair,
        x="method_label",
        y="accuracy",
        hue="split_label",
        col="budget",
        kind="bar",
        order=[METHOD_LABELS[m] for m in METHOD_ORDER],
        errorbar=("ci", 95),
        height=5,
        aspect=1.15,
        palette=["#4C78A8", "#F58518"],
    )
    g.set_axis_labels("", "Accuracy")
    g.set_titles("Pair-label budget: {col_name}")
    for ax in g.axes.flat:
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis="x", rotation=25)
    sns.move_legend(
        g,
        "upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        title="",
        frameon=False,
    )
    g.figure.tight_layout(rect=[0, 0, 1, 0.94])
    g.figure.savefig(figures_dir / "accuracy_by_method_budget.png", dpi=180)
    plt.close(g.figure)


def plot_ood_by_sum(predictions: pd.DataFrame, figures_dir: Path) -> None:
    max_budget = int(predictions["budget"].max())
    ood = predictions[(predictions["split"] == "ood_high_high") & (predictions["budget"] == max_budget)]
    grouped = (
        ood.groupby(["method", "true_sum"], as_index=False)
        .agg(accuracy=("correct", "mean"), n=("correct", "size"))
    )
    grouped = add_method_labels(grouped)
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(11, 5.5))
    sns.lineplot(
        data=grouped,
        x="true_sum",
        y="accuracy",
        hue="method_label",
        style="method_label",
        markers=True,
        dashes=False,
        hue_order=[METHOD_LABELS[m] for m in METHOD_ORDER],
    )
    plt.ylim(0, 1.02)
    plt.xlabel("True high+high sum")
    plt.ylabel("Accuracy")
    plt.title(f"OOD accuracy by sum, budget {max_budget}")
    plt.legend(title="", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout()
    plt.savefig(figures_dir / "ood_accuracy_by_sum.png", dpi=180)
    plt.close()


def plot_digit_grounding(metrics: pd.DataFrame, figures_dir: Path) -> None:
    digit = metrics[metrics["split"] == "digit_test"].copy()
    if digit.empty:
        return
    digit = add_method_labels(digit)
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=digit,
        x="budget",
        y="digit_accuracy",
        hue="method_label",
        hue_order=[METHOD_LABELS[m] for m in METHOD_ORDER if m != "direct_sum"],
        errorbar=("ci", 95),
        palette=["#54A24B", "#E45756", "#72B7B2"],
    )
    plt.ylim(0, 1.0)
    plt.xlabel("Pair-label budget")
    plt.ylabel("Digit accuracy")
    plt.title("Grounding of learned digit predicates")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(figures_dir / "digit_grounding.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/addmnist")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(results_dir / "metrics.csv")
    predictions = pd.read_csv(results_dir / "predictions.csv")

    # The prediction file intentionally omits method/budget/seed for compactness;
    # recover them from the run id, then keep all analysis keyed explicitly.
    parsed = predictions["run_id"].str.extract(r"(?P<method>.+)_b(?P<budget>\d+)_s(?P<seed>\d+)")
    predictions = pd.concat([predictions, parsed], axis=1)
    predictions["budget"] = predictions["budget"].astype(int)
    predictions["seed"] = predictions["seed"].astype(int)

    pair_summary = (
        metrics[metrics["split"].isin(["iid_restricted", "ood_high_high"])]
        .groupby(["method", "budget", "split"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            nll_mean=("nll", "mean"),
            confidence_mean=("mean_confidence", "mean"),
            train_seconds_mean=("train_seconds", "mean"),
            n_runs=("run_id", "nunique"),
        )
    )
    digit_summary = (
        metrics[metrics["split"] == "digit_test"]
        .groupby(["method", "budget"], as_index=False)
        .agg(
            digit_accuracy_mean=("digit_accuracy", "mean"),
            digit_accuracy_std=("digit_accuracy", "std"),
            digit_macro_f1_mean=("digit_macro_f1", "mean"),
            n_runs=("run_id", "nunique"),
        )
    )
    tests = paired_tests(metrics)
    boot = prediction_bootstrap(predictions)
    region_summary = ood_sum_region_summary(predictions)

    pair_summary.to_csv(results_dir / "summary_pair_metrics.csv", index=False)
    digit_summary.to_csv(results_dir / "summary_digit_metrics.csv", index=False)
    tests.to_csv(results_dir / "statistical_tests.csv", index=False)
    boot.to_csv(results_dir / "bootstrap_accuracy_ci.csv", index=False)
    region_summary.to_csv(results_dir / "ood_sum_region_summary.csv", index=False)

    plot_accuracy(metrics, figures_dir)
    plot_ood_by_sum(predictions, figures_dir)
    plot_digit_grounding(metrics, figures_dir)

    summary_payload = {
        "best_ood_by_budget": (
            pair_summary[pair_summary["split"] == "ood_high_high"]
            .sort_values(["budget", "accuracy_mean"], ascending=[True, False])
            .groupby("budget")
            .head(1)
            .to_dict(orient="records")
        ),
        "paired_tests": tests.to_dict(orient="records"),
        "ood_sum_regions": region_summary.to_dict(orient="records"),
    }
    (results_dir / "analysis_summary.json").write_text(json.dumps(summary_payload, indent=2))
    print(f"Saved summaries in {results_dir} and figures in {figures_dir}")


if __name__ == "__main__":
    main()
