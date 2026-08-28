"""
src/eval/aggregate_results.py

Pulls every results/*.json file in this repo into one table and one chart,
so a reader (or a committee) doesn't have to open 40 separate small JSON
files to see the whole picture. Also folds in the two published external
reference numbers this project should be compared against directly:
VoiceMark's own Table 3 imperceptibility metrics, and SafeSpeech's own
reported AudioPure-purification result (the number the project plan named
as "the baseline to beat" for the disruption objective) -- neither of these
previously appeared anywhere in this repo's results tables.

Does NOT invent numbers: everything under "this project" is read directly
from results/*.json; everything under "published reference" is a literal
transcription of the source paper's own reported number, clearly labeled as
such, not measured by this pipeline.

Usage:
    python src/eval/aggregate_results.py --results_dir ./results --out_dir ./results/aggregated

Outputs:
    results/aggregated/summary_table.md   -- one markdown table, human-readable
    results/aggregated/summary_table.csv  -- same data, for further analysis
    results/aggregated/summary_chart.png  -- grouped bar chart, key metrics only
"""

import os
import re
import json
import glob
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- categorical palette (validated, colorblind-safe order) -- see the
# dataviz skill's references/palette.md for the full derivation. Only the
# first few slots are used here since this chart has few series.
PALETTE = {
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "magenta": "#e87ba4", "violet": "#4a3aa7",
}
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# Published reference numbers -- literal transcriptions, not measured here.
VOICEMARK_PUBLISHED = {"pesq": 2.20, "stoi": 0.89, "si_snr_db": 2.01}
SAFESPEECH_AUDIOPURE_PUBLISHED = {
    "note": "SafeSpeech's own reported AudioPure result (WER, not ACC -- "
            "different metric family, listed for context, not a direct "
            "row-for-row diff against this project's ACC-based table).",
    "wer_before": 0.996, "wer_after": 0.857, "sim_after": 0.261,
}


def _load_all(results_dir: str) -> list:
    records = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        with open(path) as f:
            d = json.load(f)
        rec = {"file": os.path.basename(path), "label": d.get("label", ""), "checkpoint": d.get("checkpoint")}
        rec.update({k: v for k, v in d.get("results", {}).items() if not isinstance(v, list)})
        records.append(rec)
    return records


def _classify(row: dict) -> str:
    """Best-effort category tag from filename/keys, for chart grouping."""
    fname = row["file"].lower()
    if "audiopure" in fname:
        return "AudioPure purification"
    if "fpr" in fname:
        return "False positive rate"
    if "sim" in fname and "audioseal" not in fname:
        return "Disruption (SIM)"
    if "quality" in fname:
        return "Audio quality (PESQ/STOI/SI-SNR)"
    if "audioseal" in fname:
        return "AudioSeal baseline"
    if "far" in fname:
        return "FAR (attribution)"
    if "far" in fname:
        return "FAR (attribution)"
    if any(k in row for k in ("clean", "masking", "shuffling", "replacing", "neural")):
        return "Augmentation robustness"
    return "Other"


def build_table(records: list) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["category"] = df.apply(_classify, axis=1)
    preferred = ["category", "file", "label", "checkpoint"]
    other_cols = [c for c in df.columns if c not in preferred]
    return df[preferred + other_cols]


def write_table(df: pd.DataFrame, out_dir: str):
    csv_path = os.path.join(out_dir, "summary_table.csv")
    md_path = os.path.join(out_dir, "summary_table.md")
    df.to_csv(csv_path, index=False)

    lines = ["# Aggregated results summary\n",
             f"Generated from {df['file'].nunique()} files in results/. "
             "Values under 'this project' are measured; the two reference blocks "
             "below are literal transcriptions of published numbers, not measured here.\n"]
    for cat, sub in df.groupby("category"):
        lines.append(f"\n## {cat}\n")
        cols = ["label"] + [c for c in sub.columns
                             if c not in ("category", "file", "label", "checkpoint")
                             and sub[c].notna().any()]
        sub_display = sub[cols].round(4)
        lines.append(sub_display.to_markdown(index=False))
        lines.append("")

    lines.append("\n## Published reference numbers (NOT measured by this pipeline)\n")
    lines.append(f"**VoiceMark, own Table 3**: PESQ={VOICEMARK_PUBLISHED['pesq']}, "
                  f"STOI={VOICEMARK_PUBLISHED['stoi']}, SI-SNR={VOICEMARK_PUBLISHED['si_snr_db']} dB.\n")
    lines.append(f"**SafeSpeech, own AudioPure eval**: WER {SAFESPEECH_AUDIOPURE_PUBLISHED['wer_before']:.1%} -> "
                  f"{SAFESPEECH_AUDIOPURE_PUBLISHED['wer_after']:.1%} after purification, "
                  f"SIM rises to {SAFESPEECH_AUDIOPURE_PUBLISHED['sim_after']:.3f}. "
                  f"{SAFESPEECH_AUDIOPURE_PUBLISHED['note']}\n")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[write_table] Wrote {csv_path} and {md_path}")


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRIDLINE)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)


def _short_label(filename: str) -> str:
    """Derives a short, unique-ish chart label from a results/*.json filename
    (much more legible than the raw checkpoint path, and unique per file
    unlike 'label', which repeats e.g. 'baseline' across many files)."""
    name = re.sub(r"^results_", "", filename)
    name = re.sub(r"\.json$", "", name)
    return name


def build_chart(df: pd.DataFrame, out_dir: str):
    """One figure, three panels: AudioPure drop (the headline result),
    disruption SIM (the negative result, with a noise band), and quality
    metrics against VoiceMark's own published numbers."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.patch.set_facecolor(SURFACE)

    # Panel 1: AudioPure ACC before/after
    ap = df[df["category"] == "AudioPure purification"].dropna(subset=["acc_before", "acc_after"])
    ap = ap.drop_duplicates(subset=["file"])
    ax = axes[0]
    x = range(len(ap))
    labels = [_short_label(f) for f in ap["file"]]
    ax.bar([i - 0.18 for i in x], ap["acc_before"], width=0.36, color=PALETTE["blue"], label="before")
    ax.bar([i + 0.18 for i in x], ap["acc_after"], width=0.36, color=PALETTE["orange"], label="after")
    ax.axhline(0.5, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title("AudioPure: detection ACC before/after", color=INK_PRIMARY, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)

    # Panel 2: Disruption SIM
    sim = df[df["category"] == "Disruption (SIM)"].dropna(subset=["sim"]) if "sim" in df.columns else pd.DataFrame()
    ax = axes[1]
    if not sim.empty:
        sim = sim.copy()
        sim["short_label"] = sim["file"].apply(_short_label)
        agg = sim.groupby("short_label")["sim"].mean().reset_index()
        ax.bar(range(len(agg)), agg["sim"], color=PALETTE["aqua"])
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg["short_label"], rotation=40, ha="right", fontsize=7)
        ax.set_title("Disruption objective: SIM (lower = more disrupted)", color=INK_PRIMARY, fontsize=10)
    else:
        ax.text(0.5, 0.5, "no SIM data found", ha="center", va="center", color=INK_MUTED)
    _style_axes(ax)

    # Panel 3: Quality metrics vs VoiceMark's own published numbers
    q = df[df["category"] == "Audio quality (PESQ/STOI/SI-SNR)"]
    ax = axes[2]
    if not q.empty and "mean_pesq" in q.columns:
        # Prefer the largest-n baseline row (n50 measurement is the one
        # actually compared against VoiceMark's published numbers in the
        # writeup) rather than an arbitrary first row.
        baseline_rows = q[q["file"].str.contains("baseline", case=False, na=False)]
        q_best = (baseline_rows if not baseline_rows.empty else q).dropna(subset=["mean_pesq"]).iloc[-1]
        metric_pairs = [("mean_pesq", "pesq"), ("mean_stoi", "stoi"), ("mean_si_snr_db", "si_snr_db")]
        metric_pairs = [(c, k) for c, k in metric_pairs if c in q.columns]
        this_vals = [q_best.get(c, float("nan")) for c, _ in metric_pairs]
        pub_vals = [VOICEMARK_PUBLISHED.get(k, float("nan")) for _, k in metric_pairs]
        xpos = range(len(metric_pairs))
        ax.bar([i - 0.18 for i in xpos], this_vals, width=0.36, color=PALETTE["blue"], label="this project")
        ax.bar([i + 0.18 for i in xpos], pub_vals, width=0.36, color=PALETTE["yellow"], label="VoiceMark (published)")
        ax.set_xticks(list(xpos)); ax.set_xticklabels([k for _, k in metric_pairs], fontsize=8)
        ax.set_title(f"Imperceptibility vs. VoiceMark's own numbers\n(this project: {q_best['file']})",
                      color=INK_PRIMARY, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.text(0.5, 0.5, "no quality data found", ha="center", va="center", color=INK_MUTED)
    _style_axes(ax)

    for ax in axes:
        ax.set_ylabel("")

    fig.suptitle("Thesis results at a glance (auto-generated from results/*.json)",
                  color=INK_PRIMARY, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(out_dir, "summary_chart.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"[build_chart] Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--out_dir", type=str, default="./results/aggregated")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    records = _load_all(args.results_dir)
    if not records:
        print(f"[main] No JSON files found in {args.results_dir}")
        return
    df = build_table(records)
    write_table(df, args.out_dir)
    build_chart(df, args.out_dir)


if __name__ == "__main__":
    main()
