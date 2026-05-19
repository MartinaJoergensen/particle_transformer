#!/usr/bin/env python3
"""
One-click plot: best validation accuracy and test OvO AUC vs weight fractional bits (wf).
Pulls all relevant runs from W&B project par-t-quant and saves a figure.

Usage:
    cd /eos/home-m/majorgen/particle_transformer
    python3 pquant_experiments/plot_accuracy_vs_wf.py
    python3 pquant_experiments/plot_accuracy_vs_wf.py --output my_plot.pdf --csv data.csv
"""

import argparse
import re

import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY = "martina-jorgensen-cern"
PROJECT = "par-t-quant"
BASELINE_ACC = 86.05  # full-precision ParT baseline (%)


def get_wf(run):
    """Extract weight fractional bits (wf) from run config or run name."""
    wf = run.config.get("WEIGHT_F_BITS") or run.config.get("weight_f_bits")
    if wf is not None:
        return int(wf)
    # Parse from run name, e.g. "scan-wf6" → 6, "run9-i4f11" won't match
    m = re.search(r"wf(\d+)", run.name or "")
    if m:
        return int(m.group(1))
    # run9-i4f11 is the wf=14 reference run
    if run.id == "g4rpn4g0" or "run9" in (run.name or ""):
        return 14
    return None


def best_eval_acc(run):
    """Return highest eval/acc_epoch seen across all logged epochs, as a percentage."""
    try:
        rows = list(run.scan_history(keys=["eval/acc_epoch"]))
        vals = [r["eval/acc_epoch"] for r in rows if r.get("eval/acc_epoch") is not None]
        if vals:
            v = max(vals)
            return v * 100 if v <= 1.0 else v  # fraction → %
    except Exception:
        pass
    return None


def fetch_data():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    records = []
    for run in runs:
        if run.state not in ("finished", "running"):
            continue

        wf = get_wf(run)
        if wf is None:
            print(f"  [skip] run {run.id} ({run.name}): could not determine wf")
            continue

        acc = best_eval_acc(run)
        auc = run.summary.get("test/auc_ovo_macro")

        if acc is None and auc is None:
            print(f"  [skip] run {run.id} ({run.name}, wf={wf}): no metrics yet")
            continue

        print(f"  wf={wf:2d}  acc={f'{acc:.2f}%' if acc else 'n/a':8s}  auc={auc if auc else 'n/a'}  ({run.name}, {run.state})")
        records.append({
            "wf": wf,
            "run_id": run.id,
            "run_name": run.name,
            "best_eval_acc_%": acc,
            "test_auc_ovo": auc,
            "state": run.state,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # If there are duplicate wf values (e.g. restarts), keep the one with higher accuracy
    df = df.sort_values(["wf", "best_eval_acc_%"], na_position="first")
    df = df.drop_duplicates(subset="wf", keep="last")
    df = df.sort_values("wf").reset_index(drop=True)
    return df


def plot(df, output):
    has_acc = df["best_eval_acc_%"].notna().any()
    has_auc = df["test_auc_ovo"].notna().any()
    n_panels = int(has_acc) + int(has_auc)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.5), squeeze=False)
    axes = axes[0]

    panel_specs = []
    if has_acc:
        panel_specs.append(("best_eval_acc_%", "Validation accuracy (%)", "Accuracy vs weight bit width", "steelblue"))
    if has_auc:
        panel_specs.append(("test_auc_ovo", "Test OvO macro AUC", "AUC vs weight bit width", "darkorange"))

    for ax, (col, ylabel, title, color) in zip(axes, panel_specs):
        sub = df[df[col].notna()].copy()

        ax.plot(sub["wf"], sub[col], "o-", color=color, markersize=7,
                linewidth=1.8, zorder=3, label="scan runs")

        # Annotate each point with its value
        for _, row in sub.iterrows():
            val = row[col]
            label = f"{val:.2f}{'%' if col == 'best_eval_acc_%' else ''}"
            ax.annotate(label, xy=(row["wf"], val), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8, color=color)

        # Dashed baseline line on accuracy panel
        if col == "best_eval_acc_%":
            ax.axhline(BASELINE_ACC, color="grey", linestyle="--", linewidth=1.2,
                       label=f"Baseline {BASELINE_ACC}%")
            ax.legend(fontsize=9)

        ax.set_xlabel("Weight fractional bits (wf)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        if not sub.empty:
            ax.set_xticks(sorted(sub["wf"].unique()))
        ax.grid(True, linestyle="--", alpha=0.45)

    plt.suptitle("ParT compression: weight bit-width scan\n(df=11, i=4)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy and AUC vs wf from W&B")
    parser.add_argument("--output", default="acc_vs_wf.pdf", help="Output figure path")
    parser.add_argument("--csv", default=None, help="Also save data table to CSV")
    args = parser.parse_args()

    print(f"Fetching runs from W&B ({ENTITY}/{PROJECT})...\n")
    df = fetch_data()

    if df.empty:
        print("\nNo runs with data found. Make sure you are logged in: wandb login")
        return

    print("\n--- Summary table ---")
    print(df[["wf", "run_name", "best_eval_acc_%", "test_auc_ovo", "state"]].to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nTable saved to: {args.csv}")

    plot(df, args.output)


if __name__ == "__main__":
    main()
