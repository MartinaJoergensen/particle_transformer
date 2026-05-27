#!/usr/bin/env python3
"""
One-click plot: test accuracy and OvO AUC vs weight fractional bits (wf).
Pulls all relevant runs from W&B project par-t-quant and saves TWO separate figures.

Usage:
    cd /eos/home-m/majorgen/particle_transformer
    python3 pquant_experiments/plot_accuracy_vs_wf.py
    python3 pquant_experiments/plot_accuracy_vs_wf.py \
        --output-acc acc_vs_wf.pdf --output-auc auc_vs_wf.pdf --csv data.csv
"""

import argparse
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import wandb

ENTITY       = "martina-jorgensen-cern"
PROJECT      = "par-t-quant"
BASELINE_ACC = 86.05   # full-precision ParT baseline (%)

# ── W&B-inspired light palette ───────────────────────────────────────────────
BG       = "#ffffff"   # white figure / axes background
BLUE     = "#0073E6"   # W&B blue — main line / marker colour
TEXT     = "#4a5568"   # mid-grey labels (W&B style)
GRID     = "#e8edf2"   # faint horizontal grid
GREY_DIM = "#9aacbb"   # captions / secondary text

mpl.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    BG,
    "savefig.facecolor": BG,
    "text.color":        TEXT,
    "axes.labelcolor":   TEXT,
    "xtick.color":       TEXT,
    "ytick.color":       TEXT,
    "axes.edgecolor":    GRID,
    "grid.color":        GRID,
    "legend.facecolor":  BG,
    "legend.edgecolor":  GRID,
    "legend.labelcolor": TEXT,
    "font.family":       "sans-serif",
})


def get_wf(run):
    """Extract weight fractional bits (wf) from run config or name."""
    wf = run.config.get("WEIGHT_F_BITS") or run.config.get("weight_f_bits")
    if wf is not None:
        return int(wf)
    m = re.search(r"wf(\d+)", run.name or "")
    if m:
        return int(m.group(1))
    return None


def eval_acc_from_summary(run):
    """Return eval/acc_epoch from run summary (last epoch), as a percentage."""
    v = run.summary.get("eval/acc_epoch")
    if v is not None:
        return float(v) * 100 if float(v) <= 1.0 else float(v)
    return None


def fetch_data():
    api  = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    records = []
    for run in runs:
        if run.state not in ("finished", "running", "crashed"):
            continue

        wf = get_wf(run)
        if wf is None or wf == 14:   # skip run9 25-epoch pretest
            continue

        test_acc = run.summary.get("test/acc")
        auc      = run.summary.get("test/auc_ovo_macro")
        eval_acc = eval_acc_from_summary(run)

        acc         = test_acc if test_acc is not None else eval_acc
        acc_is_test = test_acc is not None

        if acc is None and auc is None:
            print(f"  [skip] run {run.id} ({run.name}, wf={wf}): no metrics yet")
            continue

        src = "test" if acc_is_test else "eval/last"
        print(f"  wf={wf:2d}  acc={f'{acc:.3f}%' if acc else 'n/a':9s}({src})"
              f"  auc={auc if auc else 'n/a'}  ({run.name}, {run.state})")
        records.append({
            "wf":          wf,
            "run_id":      run.id,
            "run_name":    run.name,
            "acc_%":       acc,
            "acc_is_test": acc_is_test,
            "test_auc_ovo": auc,
            "state":       run.state,
        })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["wf", "acc_%"], na_position="first")
    df = df.drop_duplicates(subset="wf", keep="last")
    df = df.sort_values("wf").reset_index(drop=True)
    return df


FRAME_CLR = "#d0d7df"   # light border around plot area

def _style_ax(ax):
    """W&B-like style: light frame border, no tick marks, horizontal-only grid."""
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(FRAME_CLR)
        sp.set_linewidth(0.8)
    ax.tick_params(colors=TEXT, labelsize=10, length=0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.8, alpha=0.7, color=GRID)
    ax.grid(False, axis="x")


def plot_single(df, col, ylabel, title, output, baseline=None,
                tested_col="acc_is_test"):
    """Produce one W&B-style plot for `col` and save to `output`."""
    sub = df[df[col].notna()].copy()
    if sub.empty:
        print(f"  [skip] no data for {col}, skipping {output}")
        return

    tested   = sub[sub[tested_col] == True]
    untested = sub[sub[tested_col] == False]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(sub["wf"], sub[col], "-", color=BLUE, linewidth=2.2, zorder=2, alpha=0.9)
    if not tested.empty:
        ax.plot(tested["wf"], tested[col], "o",
                color=BLUE, markersize=8, zorder=3, label="test result")
    if not untested.empty:
        ax.plot(untested["wf"], untested[col], "o",
                color=BLUE, markersize=8, zorder=3,
                markerfacecolor=BG, markeredgewidth=2.0,
                label="val acc epoch 39 (crashed — f=2 too coarse)")

    for _, row in sub.iterrows():
        val = row[col]
        lbl = f"{val:.3f}%" if col == "acc_%" else f"{val:.5f}"
        ax.annotate(lbl, xy=(row["wf"], val), xytext=(0, 10),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, color=BLUE, fontweight="bold")

    if baseline is not None:
        ax.axhline(baseline, color=GREY_DIM, linestyle="--", linewidth=1.2,
                   alpha=0.8, label=f"FP baseline  {baseline}%", zorder=1)

    _style_ax(ax)
    ax.set_xlabel("Weight fractional bits (wf)", fontsize=12, color=TEXT, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12, color=TEXT, labelpad=8)
    ax.set_title(title, fontsize=13, color=TEXT, fontweight="bold", pad=12)
    ax.set_xticks(sorted(sub["wf"].unique()))

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.10)

    if baseline is not None or not untested.empty:
        ax.legend(fontsize=9, framealpha=0.8, loc="lower right")

    fig.text(0.5, 0.01,
             "Data compressed to 16 bits (f=11, i=4, k=1)",
             ha="center", fontsize=8.5, color=GREY_DIM, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(output, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved → {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot accuracy and AUC vs wf from W&B (two files)")
    parser.add_argument("--output-acc", default="acc_vs_wf.pdf")
    parser.add_argument("--output-auc", default="auc_vs_wf.pdf")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()

    print(f"Fetching runs from W&B ({ENTITY}/{PROJECT})...\n")
    df = fetch_data()

    if df.empty:
        print("\nNo runs with data found. Make sure you are logged in: wandb login")
        return

    print("\n--- Summary table ---")
    print(df[["wf", "run_name", "acc_%", "acc_is_test", "test_auc_ovo", "state"]].to_string(index=False))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nTable saved to: {args.csv}")

    print()
    plot_single(df, "acc_%",
                ylabel="Accuracy (%)",
                title="ParT weights compression — accuracy vs wf bits",
                output=args.output_acc,
                baseline=BASELINE_ACC)
    plot_single(df, "test_auc_ovo",
                ylabel="Test OvO macro AUC",
                title="ParT weights compression — OvO AUC vs wf bits",
                output=args.output_auc)


if __name__ == "__main__":
    main()
