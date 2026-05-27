#!/usr/bin/env python3
"""
make_slides.py — Clean light-theme slide deck for ParT wf scan results.

Outputs:
  scan_slides.pdf  — 5 slides: accuracy plot, AUC plot, overview table,
                     per-class AUC table, QCD rejection table.

Usage:
    cd /eos/home-m/majorgen/particle_transformer
    python3 pquant_experiments/make_slides.py
    python3 pquant_experiments/make_slides.py --slides scan_slides.pdf
"""

import argparse
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import wandb

mpl.rcParams["font.family"] = "sans-serif"

# ── Project constants ─────────────────────────────────────────────────────────
ENTITY       = "martina-jorgensen-cern"
PROJECT      = "par-t-quant"
BASELINE_ACC = 86.05
BASELINE_RW  = 1.0

CLASSES        = ['QCD', 'Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
SIGNAL_CLASSES = ['Hbb', 'Hcc', 'Hgg', 'H4q', 'Hqql', 'Zqq', 'Wqq', 'Tbqq', 'Tbl']
TPR_TARGETS    = {
    'Hbb': 0.50, 'Hcc': 0.50, 'Hgg': 0.50, 'H4q': 0.50,
    'Hqql': 0.99, 'Zqq': 0.50, 'Wqq': 0.50, 'Tbqq': 0.50, 'Tbl': 0.995,
}

# Full-precision ParT baseline — metrics from pred.root + paper
BASELINE = {
    "wf":           "FP",
    "run_name":     "full-precision",
    "acc_%":        86.051,     # computed: 86.0507 %
    "acc_is_test":  True,
    "remaining_w":  1.0,
    "test_auc_ovo": 0.987720,   # OvO macro AUC (0.9877199…)
    # Per-class OvR AUC — computed from baseline/pred.root
    "auc_QCD":   0.978251,
    "auc_Hbb":   0.997134,
    "auc_Hcc":   0.987901,
    "auc_Hgg":   0.980071,
    "auc_H4q":   0.988796,
    "auc_Hqql":  0.999567,
    "auc_Zqq":   0.966947,
    "auc_Wqq":   0.980131,
    "auc_Tbqq":  0.998537,
    "auc_Tbl":   0.999865,
    # QCD rejection rates from paper (consistent with our pipeline)
    "rej_Hbb":   10638,
    "rej_Hcc":   4149,
    "rej_Hgg":   123,
    "rej_H4q":   1867,
    "rej_Hqql":  5420,
    "rej_Zqq":   402,
    "rej_Wqq":   543,
    "rej_Tbqq":  32258,
    "rej_Tbl":   16260,
}

# ── Light slide palette ───────────────────────────────────────────────────────
SLIDE_BG  = "#ffffff"   # white
BLUE      = "#0073E6"   # W&B-style blue — main accent
TEXT      = "#1a2233"   # dark title text
SUBTEXT   = "#4a5568"   # axis labels / body text
GREY      = "#9aacbb"   # captions / secondary
GRID      = "#e8edf2"   # plot grid lines
FRAME_CLR = "#d0d7df"   # border around plot area
HDR_BG    = "#dbeafe"   # table header light blue
HDR_TEXT  = "#1e40af"   # table header dark blue text
ROW_A     = "#ffffff"   # white row
ROW_B     = "#f1f5f9"   # very-light-grey row
BORDER    = "#e2e8f0"   # cell borders

SLIDE_W, SLIDE_H = 13.33, 7.5   # 16:9 inches

# Margin in figure-fraction units (~1.5 cm on 13.33 × 7.5 in slide)
MX = 0.048   # horizontal margin fraction
MY = 0.082   # vertical margin fraction


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_wf(run):
    wf = run.config.get("WEIGHT_F_BITS") or run.config.get("weight_f_bits")
    if wf is not None:
        return int(wf)
    m = re.search(r"wf(\d+)", run.name or "")
    if m:
        return int(m.group(1))
    return None


def _pct(v):
    if v is None:
        return None
    return float(v) * 100 if float(v) <= 1.0 else float(v)


def _is_nan(v):
    return v is None or (isinstance(v, float) and np.isnan(v))


def fmt_acc(v):
    return f"{v:.3f}%" if not _is_nan(v) else "—"

def fmt_auc(v):
    return f"{v:.5f}" if not _is_nan(v) else "—"

def fmt_wt(v):
    if _is_nan(v):
        return "—"
    p = float(v) * 100 if float(v) <= 1.0 else float(v)
    return f"{p:.2f}%"

def fmt_rej(v):
    if _is_nan(v):
        return "—"
    return "∞" if v == float("inf") else f"{v:.3f}"


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_full_data():
    api  = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    records = []
    for run in runs:
        if run.state not in ("finished", "running", "crashed"):
            continue

        wf = get_wf(run)
        if wf is None or wf == 14:
            continue

        s        = run.summary
        test_acc = s.get("test/acc")
        auc_ovo  = s.get("test/auc_ovo_macro")
        eval_acc = _pct(s.get("eval/acc_epoch"))
        rem_w    = s.get("eval/remaining_weights")

        acc         = test_acc if test_acc is not None else eval_acc
        acc_is_test = test_acc is not None

        if _is_nan(acc) and _is_nan(auc_ovo):
            continue

        row = {
            "wf":           wf,
            "run_name":     run.name,
            "acc_%":        acc,
            "acc_is_test":  acc_is_test,
            "remaining_w":  rem_w,
            "test_auc_ovo": auc_ovo,
            "state":        run.state,
        }
        for cls in CLASSES:
            row[f"auc_{cls}"] = s.get(f"test/auc_{cls}")
        for cls in SIGNAL_CLASSES:
            lbl = int(TPR_TARGETS[cls] * 100)
            row[f"rej_{cls}"] = s.get(f"test/rejection_{cls}_at{lbl}eff")

        records.append(row)
        src = "test" if acc_is_test else "eval/last"
        print(f"  wf={wf:2d}  acc={fmt_acc(acc)}({src})  auc={fmt_auc(auc_ovo)}"
              f"  ({run.name}, {run.state})")

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.sort_values(["wf", "acc_%"], na_position="first")
    df = df.drop_duplicates(subset="wf", keep="last")
    df = df.sort_values("wf").reset_index(drop=True)
    return df


# ── Slide helpers ─────────────────────────────────────────────────────────────

def blank_slide():
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H))
    fig.patch.set_facecolor(SLIDE_BG)
    return fig


def add_header(fig, title, line2=None):
    """Blue left stripe, bold dark title, blue second line."""
    # Left stripe — stays at full height, thin
    stripe = fig.add_axes([0, 0, 0.006, 1])
    stripe.set_facecolor(BLUE)
    stripe.axis("off")

    # Title
    fig.text(0.006 + MX, 1 - MY, title,
             color=TEXT, fontsize=20, fontweight="bold",
             va="top", ha="left", transform=fig.transFigure)
    # Second header line
    if line2:
        fig.text(0.006 + MX, 1 - MY - 0.10, line2,
                 color=BLUE, fontsize=11,
                 va="top", ha="left", transform=fig.transFigure)


def _style_plot_ax(ax):
    """W&B-like axes: visible frame border, no tick marks, horizontal-only grid."""
    # Light frame around axes area
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_color(FRAME_CLR)
        sp.set_linewidth(1.0)
    ax.tick_params(colors=SUBTEXT, labelsize=10, length=0)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.8, alpha=0.65, color=GRID)
    ax.grid(False, axis="x")


def draw_plot_on_slide(fig, df, col, ylabel, baseline_val=None,
                       tested_col="acc_is_test"):
    sub = df[df[col].notna()].copy()
    if sub.empty:
        return

    # Plot area: leave header space at top (~28% of height) + margins
    ax_left   = 0.006 + MX + 0.03
    ax_bottom = MY + 0.01
    ax_width  = 1 - ax_left - MX - 0.03
    ax_height = 0.70 - 2 * MY
    ax = fig.add_axes([ax_left, ax_bottom, ax_width, ax_height])
    ax.set_facecolor(SLIDE_BG)

    tested   = sub[sub[tested_col] == True]
    untested = sub[sub[tested_col] == False]

    ax.plot(sub["wf"], sub[col], "-", color=BLUE, linewidth=2.2, zorder=2, alpha=0.9)
    if not tested.empty:
        ax.plot(tested["wf"], tested[col], "o",
                color=BLUE, markersize=8, zorder=3, label="test result")
    if not untested.empty:
        ax.plot(untested["wf"], untested[col], "o",
                color=BLUE, markersize=8, zorder=3,
                markerfacecolor=SLIDE_BG, markeredgewidth=2.0,
                label="val acc epoch 39 (crashed — f=2 too coarse)")

    for _, row in sub.iterrows():
        val = row[col]
        lbl = f"{val:.3f}%" if col == "acc_%" else f"{val:.5f}"
        ax.annotate(lbl, xy=(row["wf"], val), xytext=(0, 11),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, color=BLUE, fontweight="bold")

    if baseline_val is not None:
        ax.axhline(baseline_val, color=GREY, linestyle="--", linewidth=1.3,
                   alpha=0.9,
                   label=f"FP baseline  {baseline_val}", zorder=1)

    _style_plot_ax(ax)
    ax.set_xlabel("Weight fractional bits (wf)", fontsize=12, color=SUBTEXT, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=12, color=SUBTEXT, labelpad=6)
    ax.set_xticks(sorted(sub["wf"].unique()))

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + (ymax - ymin) * 0.14)

    if baseline_val is not None or not untested.empty:
        ax.legend(fontsize=9, framealpha=0.9, loc="lower right",
                  edgecolor=FRAME_CLR, labelcolor=SUBTEXT)


def draw_table(ax, cell_text, col_labels, row_labels, note=None):
    """Light-theme table; first data row = baseline (highlighted blue)."""
    ax.axis("off")
    ax.set_facecolor(SLIDE_BG)

    n_rows = len(cell_text)
    n_cols = len(col_labels)

    cell_colors = []
    for i in range(n_rows):
        if i == 0:                          # baseline row — light blue tint
            cell_colors.append(["#eff6ff"] * n_cols)
        elif i % 2 == 0:
            cell_colors.append([ROW_A] * n_cols)
        else:
            cell_colors.append([ROW_B] * n_cols)

    row_label_colors = ["#eff6ff"] + [ROW_A if i % 2 == 0 else ROW_B
                                       for i in range(1, n_rows)]

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellColours=cell_colors,
        colColours=[HDR_BG] * n_cols,
        rowColours=row_label_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.65)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(BORDER)
        cell.set_linewidth(0.5)
        if r == 0:          # column header
            cell.set_text_props(color=HDR_TEXT, fontweight="bold", fontsize=8.5)
        elif c == -1:       # row label
            color = BLUE if r == 1 else SUBTEXT
            cell.set_text_props(color=color, fontweight="bold")
        else:
            color = BLUE if r == 1 else TEXT
            bold  = (r == 1)
            cell.set_text_props(color=color, fontweight="bold" if bold else "normal")

    if note:
        ax.text(0.5, 0.0, note, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7, color=GREY, style="italic")


# ── Individual slide builders ─────────────────────────────────────────────────

PLOT_SUBTITLE = "Data compressed to 16 bits (f=11, i=4, k=1)"

def slide_accuracy(pdf, df):
    fig = blank_slide()
    add_header(fig, "ParT weights compression — accuracy vs wf bits", PLOT_SUBTITLE)
    draw_plot_on_slide(fig, df, "acc_%",
                       ylabel="Accuracy (%)",
                       baseline_val=BASELINE_ACC)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Slide 1: accuracy plot")


def slide_auc(pdf, df):
    fig = blank_slide()
    add_header(fig, "ParT weights compression — OvO AUC vs wf bits", PLOT_SUBTITLE)
    draw_plot_on_slide(fig, df, "test_auc_ovo",
                       ylabel="Test OvO macro AUC",
                       baseline_val=BASELINE["test_auc_ovo"])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Slide 2: AUC plot")


def slide_overview_table(pdf, df):
    rows_data  = [BASELINE] + df.to_dict("records")
    row_labels = ["Full prec."] + [f"wf = {r['wf']}" for r in df.to_dict("records")]
    col_labels = ["Accuracy (%)", "Remaining weights", "OvO macro AUC"]

    rows = []
    for r in rows_data:
        acc_str = fmt_acc(r["acc_%"])
        if not r["acc_is_test"]:
            acc_str += " *"
        rows.append([acc_str, fmt_wt(r["remaining_w"]), fmt_auc(r["test_auc_ovo"])])

    fig = blank_slide()
    add_header(fig, "Overall scan metrics",
               "Test accuracy  ·  remaining weights  ·  OvO macro AUC")
    ax = fig.add_axes([0.006 + MX, MY + 0.01, 1 - 2*MX - 0.006, 0.60])
    draw_table(ax, rows, col_labels, row_labels,
               note=("* wf=2: training ran to epoch 39 but weights collapsed to 0% by epoch 3 "
                     "(2-bit quantization too coarse); eval accuracy shown is 10% (random chance for 10 classes)"))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Slide 3: overview table")


def slide_auc_table(pdf, df):
    rows_data  = [BASELINE] + df.to_dict("records")
    row_labels = ["Full prec."] + [f"wf = {r['wf']}" for r in df.to_dict("records")]
    col_labels = CLASSES

    rows = []
    for r in rows_data:
        rows.append([fmt_auc(r.get(f"auc_{c}")) for c in CLASSES])

    fig = blank_slide()
    add_header(fig, "Per-class AUC (OvR)",
               "Binary one-vs-rest AUC per jet class")
    ax = fig.add_axes([MX, MY, 1 - 2*MX, 0.63])
    draw_table(ax, rows, col_labels, row_labels)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Slide 4: per-class AUC table")


def slide_rejection_table(pdf, df):
    rows_data  = [BASELINE] + df.to_dict("records")
    row_labels = ["Full prec."] + [f"wf = {r['wf']}" for r in df.to_dict("records")]

    col_labels = [f"{cls}\n@{int(TPR_TARGETS[cls]*100)}%"
                  for cls in SIGNAL_CLASSES]

    rows = []
    for r in rows_data:
        rows.append([fmt_rej(r.get(f"rej_{c}")) for c in SIGNAL_CLASSES])

    fig = blank_slide()
    add_header(fig, "QCD rejection rate",
               "1 / FPR vs QCD at per-class signal efficiency target")
    ax = fig.add_axes([MX, MY, 1 - 2*MX, 0.63])
    draw_table(ax, rows, col_labels, row_labels,
               note=("Rejection = 1 / false positive rate, where FPR = fraction of QCD background jets passing the "
                     "signal-score threshold at the given signal efficiency. "
                     "Full prec. (FP) = full-precision uncompressed ParT model, evaluated on the 20M jet test set."))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    print("  Slide 5: rejection rates table")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate clean light-theme slide deck for ParT wf scan")
    parser.add_argument("--slides", default="scan_slides.pdf")
    args = parser.parse_args()

    print(f"Fetching runs from W&B ({ENTITY}/{PROJECT})...\n")
    df = fetch_full_data()

    if df.empty:
        print("\nNo data found. Check: wandb login")
        return

    print(f"\n--- Summary ---")
    print(df[["wf", "run_name", "acc_%", "test_auc_ovo", "state"]].to_string(index=False))

    print(f"\nBuilding slides → {args.slides}")
    with PdfPages(args.slides) as pdf:
        slide_accuracy(pdf, df)
        slide_auc(pdf, df)
        slide_overview_table(pdf, df)
        slide_auc_table(pdf, df)
        slide_rejection_table(pdf, df)

    print(f"\nDone — {args.slides}  (5 slides)")


if __name__ == "__main__":
    main()
