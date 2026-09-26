#!/usr/bin/env python3
"""
===============================================================================
 Comprehensive EDA — Business Entity Resolution Challenge
===============================================================================
 Analyzes:
   • Train sources (S1, S2, S3) and ground truth
   • Test sources (S1, S2, S3)
 Produces:
   • Console summary statistics
   • 12+ publication-quality PNG charts saved to  eda_output/
===============================================================================
"""

import os
import re
import sys
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE, "dataset", "train")
TEST_DIR  = os.path.join(BASE, "dataset", "test")
OUT_DIR   = os.path.join(BASE, "eda_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#161b22",
    "axes.edgecolor":   "#30363d",
    "axes.labelcolor":  "#c9d1d9",
    "xtick.color":      "#8b949e",
    "ytick.color":      "#8b949e",
    "text.color":       "#c9d1d9",
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "grid.color":       "#21262d",
    "grid.alpha":       0.6,
})
PALETTE = ["#58a6ff", "#f78166", "#7ee787", "#d2a8ff", "#ffa657",
           "#ff7b72", "#79c0ff", "#56d364", "#e3b341", "#bc8cff"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def load(path: str) -> pd.DataFrame:
    """Load a TSV, gracefully handling encoding issues."""
    return pd.read_csv(path, sep="\t", engine="python",
                       encoding="utf-8", on_bad_lines="skip")

def save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Saved {path}")

def hr(title: str):
    print(f"\n{'═'*70}\n  {title}\n{'═'*70}")

# ══════════════════════════════════════════════════════════════════════════════
#  1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
hr("1 · Loading datasets")

train_s1 = load(os.path.join(TRAIN_DIR, "train_source1.tsv"))
train_s2 = load(os.path.join(TRAIN_DIR, "train_source2.tsv"))
train_s3 = load(os.path.join(TRAIN_DIR, "train_source3.tsv"))
gt       = load(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))

test_s1  = load(os.path.join(TEST_DIR, "test_source1.tsv"))
test_s2  = load(os.path.join(TEST_DIR, "test_source2.tsv"))
test_s3  = load(os.path.join(TEST_DIR, "test_source3.tsv"))

sources = {
    "Train S1": train_s1, "Train S2": train_s2, "Train S3": train_s3,
    "Test S1":  test_s1,  "Test S2":  test_s2,  "Test S3":  test_s3,
}
print("  All datasets loaded ✓")

# ══════════════════════════════════════════════════════════════════════════════
#  2. BASIC SHAPE & SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
hr("2 · Dataset shapes & column info")
for name, df in sources.items():
    print(f"\n  {name}:  {df.shape[0]:>10,} rows  ×  {df.shape[1]} cols  →  {list(df.columns)}")
print(f"\n  Ground Truth:  {gt.shape[0]:>10,} rows  ×  {gt.shape[1]} cols  →  {list(gt.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
#  3. MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
hr("3 · Missing values")
missing_data = []
for name, df in sources.items():
    for col in df.columns:
        n_miss = df[col].isna().sum()
        pct    = 100 * n_miss / len(df)
        missing_data.append({"Source": name, "Column": col,
                             "Missing": n_miss, "% Missing": round(pct, 2)})
        if n_miss > 0:
            print(f"  {name:12s} | {col:20s} | {n_miss:>10,} ({pct:.2f}%)")

missing_df = pd.DataFrame(missing_data)

# ── Chart: missing-value heatmap ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
pivot = missing_df.pivot(index="Source", columns="Column", values="% Missing")
pivot = pivot.fillna(0)
im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        val = pivot.values[i, j]
        color = "white" if val > 15 else "#c9d1d9"
        ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                fontsize=10, color=color, weight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("% Missing", color="#c9d1d9")
cbar.ax.yaxis.set_tick_params(color="#8b949e")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
ax.set_title("Missing Values Heatmap (% per column)")
fig.tight_layout()
save(fig, "01_missing_values_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
#  4. COUNTRY DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════
hr("4 · Country distribution")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for idx, (name, df) in enumerate(sources.items()):
    ax = axes[idx // 3][idx % 3]
    counts = df["country"].value_counts()
    print(f"\n  {name}:")
    for c, n in counts.items():
        print(f"    {c:>10s}  →  {n:>10,}  ({100*n/len(df):.1f}%)")
    colors_sel = PALETTE[:len(counts)]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=colors_sel, textprops={"color": "#c9d1d9", "fontsize": 9},
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5},
        startangle=90)
    for t in autotexts:
        t.set_fontsize(8)
        t.set_color("white")
    ax.set_title(name, fontsize=12)
fig.suptitle("Country Distribution Across Sources", fontsize=16, y=1.01, weight="bold")
fig.tight_layout()
save(fig, "02_country_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
#  5. RECORD COUNTS COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
hr("5 · Record counts comparison (Train vs Test)")
labels = ["Source 1", "Source 2", "Source 3"]
train_counts = [len(train_s1), len(train_s2), len(train_s3)]
test_counts  = [len(test_s1),  len(test_s2),  len(test_s3)]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(labels))
w = 0.32
bars1 = ax.bar(x - w/2, train_counts, w, label="Train", color=PALETTE[0], edgecolor="#0d1117")
bars2 = ax.bar(x + w/2, test_counts,  w, label="Test",  color=PALETTE[1], edgecolor="#0d1117")
for b in bars1:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5000,
            f"{b.get_height():,.0f}", ha="center", va="bottom", fontsize=9, color=PALETTE[0])
for b in bars2:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 5000,
            f"{b.get_height():,.0f}", ha="center", va="bottom", fontsize=9, color=PALETTE[1])
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Number of Records")
ax.set_title("Record Counts: Train vs Test")
ax.legend()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "03_record_counts.png")

# ══════════════════════════════════════════════════════════════════════════════
#  6. BUSINESS NAME LENGTH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
hr("6 · Business name length analysis")
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for idx, (name, df) in enumerate(sources.items()):
    ax = axes[idx // 3][idx % 3]
    lengths = df["business_name"].dropna().str.len()
    ax.hist(lengths, bins=80, color=PALETTE[idx], alpha=0.85, edgecolor="#0d1117")
    ax.axvline(lengths.median(), color="#ffa657", ls="--", lw=1.5, label=f"Median={lengths.median():.0f}")
    ax.axvline(lengths.mean(),   color="#ff7b72", ls=":",  lw=1.5, label=f"Mean={lengths.mean():.1f}")
    ax.set_title(name)
    ax.set_xlabel("Character length")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(lengths.quantile(0.99), 200))
    print(f"  {name:12s}  min={lengths.min():.0f}  max={lengths.max():.0f}  "
          f"mean={lengths.mean():.1f}  median={lengths.median():.0f}  std={lengths.std():.1f}")
fig.suptitle("Business Name Length Distribution", fontsize=16, y=1.01, weight="bold")
fig.tight_layout()
save(fig, "04_name_length_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
#  7. BUSINESS ADDRESS LENGTH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
hr("7 · Business address length analysis")
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for idx, (name, df) in enumerate(sources.items()):
    ax = axes[idx // 3][idx % 3]
    lengths = df["business_address"].dropna().str.len()
    ax.hist(lengths, bins=80, color=PALETTE[idx], alpha=0.85, edgecolor="#0d1117")
    ax.axvline(lengths.median(), color="#ffa657", ls="--", lw=1.5, label=f"Median={lengths.median():.0f}")
    ax.axvline(lengths.mean(),   color="#ff7b72", ls=":",  lw=1.5, label=f"Mean={lengths.mean():.1f}")
    ax.set_title(name)
    ax.set_xlabel("Character length")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.set_xlim(0, min(lengths.quantile(0.99), 300))
    print(f"  {name:12s}  min={lengths.min():.0f}  max={lengths.max():.0f}  "
          f"mean={lengths.mean():.1f}  median={lengths.median():.0f}  std={lengths.std():.1f}")
fig.suptitle("Business Address Length Distribution", fontsize=16, y=1.01, weight="bold")
fig.tight_layout()
save(fig, "05_address_length_distribution.png")

# ══════════════════════════════════════════════════════════════════════════════
#  8. WORD COUNT DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
hr("8 · Word-count distributions (name & address)")
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for idx, (name, df) in enumerate(sources.items()):
    ax = axes[idx // 3][idx % 3]
    name_wc = df["business_name"].dropna().str.split().str.len()
    addr_wc = df["business_address"].dropna().str.split().str.len()
    ax.hist(name_wc, bins=range(0, 20), alpha=0.7, label="Name words", color=PALETTE[0], edgecolor="#0d1117")
    ax.hist(addr_wc, bins=range(0, 25), alpha=0.5, label="Address words", color=PALETTE[2], edgecolor="#0d1117")
    ax.set_title(name)
    ax.set_xlabel("Word count")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    print(f"  {name:12s}  Name words: mean={name_wc.mean():.1f} median={name_wc.median():.0f}  |  "
          f"Addr words: mean={addr_wc.mean():.1f} median={addr_wc.median():.0f}")
fig.suptitle("Word Count: Business Name vs Address", fontsize=16, y=1.01, weight="bold")
fig.tight_layout()
save(fig, "06_word_count_distributions.png")

# ══════════════════════════════════════════════════════════════════════════════
#  9. GROUND TRUTH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
hr("9 · Ground truth analysis")

# Parse matched IDs
gt["match_list"] = gt["matched_entity_ids"].apply(
    lambda x: str(x).split(",") if pd.notna(x) and str(x).strip() != "" else []
)
gt["num_matches"]  = gt["match_list"].apply(len)
gt["has_s2"]       = gt["match_list"].apply(lambda lst: any(i.startswith("S2-") for i in lst))
gt["has_s3"]       = gt["match_list"].apply(lambda lst: any(i.startswith("S3-") for i in lst))
gt["num_s2"]       = gt["match_list"].apply(lambda lst: sum(1 for i in lst if i.startswith("S2-")))
gt["num_s3"]       = gt["match_list"].apply(lambda lst: sum(1 for i in lst if i.startswith("S3-")))

total_s1       = len(gt)
singletons     = (gt["num_matches"] == 0).sum()
with_matches   = (gt["num_matches"] > 0).sum()
both_sources   = (gt["has_s2"] & gt["has_s3"]).sum()
s2_only        = (gt["has_s2"] & ~gt["has_s3"]).sum()
s3_only        = (~gt["has_s2"] & gt["has_s3"]).sum()

print(f"  Total S1 entities:              {total_s1:>10,}")
print(f"  Singletons (no match):          {singletons:>10,}  ({100*singletons/total_s1:.1f}%)")
print(f"  With ≥1 match:                  {with_matches:>10,}  ({100*with_matches/total_s1:.1f}%)")
print(f"  Matches in both S2 & S3:        {both_sources:>10,}")
print(f"  Matches only in S2:             {s2_only:>10,}")
print(f"  Matches only in S3:             {s3_only:>10,}")
print(f"  Mean matches per entity:        {gt['num_matches'].mean():.2f}")
print(f"  Median matches per entity:      {gt['num_matches'].median():.0f}")
print(f"  Max matches per entity:         {gt['num_matches'].max()}")
print(f"  Mean S2 matches per entity:     {gt['num_s2'].mean():.2f}")
print(f"  Mean S3 matches per entity:     {gt['num_s3'].mean():.2f}")

# ── Chart: match count distribution ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 9a. Histogram of total matches per entity
ax = axes[0]
match_counts = gt["num_matches"].value_counts().sort_index()
ax.bar(match_counts.index, match_counts.values, color=PALETTE[0], edgecolor="#0d1117")
ax.set_xlabel("Number of matches per S1 entity")
ax.set_ylabel("Frequency")
ax.set_title("Match Count Distribution")
ax.set_xlim(-0.5, min(match_counts.index.max(), 15) + 0.5)
ax.grid(axis="y", alpha=0.3)

# 9b. Singleton vs matched pie
ax = axes[1]
ax.pie([singletons, with_matches],
       labels=["Singletons", "Matched"],
       autopct="%1.1f%%",
       colors=[PALETTE[5], PALETTE[2]],
       textprops={"color": "#c9d1d9"},
       wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5},
       startangle=90)
ax.set_title("Singletons vs Matched Entities")

# 9c. Source breakdown
ax = axes[2]
ax.pie([both_sources, s2_only, s3_only],
       labels=["Both S2 & S3", "S2 only", "S3 only"],
       autopct="%1.1f%%",
       colors=[PALETTE[3], PALETTE[0], PALETTE[1]],
       textprops={"color": "#c9d1d9"},
       wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5},
       startangle=90)
ax.set_title("Source Breakdown of Matched Entities")

fig.suptitle("Ground Truth — Match Statistics", fontsize=16, y=1.02, weight="bold")
fig.tight_layout()
save(fig, "07_ground_truth_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
#  10. S2 vs S3 MATCH COUNTS SCATTER
# ══════════════════════════════════════════════════════════════════════════════
hr("10 · S2 vs S3 matches per entity")
fig, ax = plt.subplots(figsize=(8, 7))
# jitter for visibility
jitter = 0.15
ax.scatter(
    gt["num_s2"] + np.random.uniform(-jitter, jitter, len(gt)),
    gt["num_s3"] + np.random.uniform(-jitter, jitter, len(gt)),
    alpha=0.05, s=3, color=PALETTE[3]
)
ax.set_xlabel("Number of S2 matches")
ax.set_ylabel("Number of S3 matches")
ax.set_title("S2 vs S3 Match Counts per S1 Entity")
ax.grid(alpha=0.2)
ax.set_xlim(-0.5, gt["num_s2"].quantile(0.99) + 1)
ax.set_ylim(-0.5, gt["num_s3"].quantile(0.99) + 1)
fig.tight_layout()
save(fig, "08_s2_vs_s3_matches.png")

# ══════════════════════════════════════════════════════════════════════════════
#  11. TEXT PATTERN ANALYSIS — COMMON SUFFIXES / ABBREVIATIONS
# ══════════════════════════════════════════════════════════════════════════════
hr("11 · Common business name patterns / suffixes")

suffix_patterns = {
    "LLC":          r"\bLLC\b",
    "Inc":          r"\bInc\b\.?",
    "Corp":         r"\bCorp\b\.?",
    "Ltd":          r"\bLtd\b\.?",
    "Limited":      r"\bLimited\b",
    "Pvt":          r"\bPvt\b\.?",
    "Private":      r"\bPrivate\b",
    "LLP":          r"\bLLP\b",
    "Co":           r"\bCo\b\.?",
    "Foundation":   r"\bFoundation\b",
    "Association":  r"\bAssociation\b",
    "Trust":        r"\bTrust\b",
    "Enterprise":   r"\bEnterprise[s]?\b",
    "Services":     r"\bServices?\b",
    "DBA":          r"\bDBA\b",
    "& / and":      r"\b(?:&|and)\b",
}

suffix_results = {}
for name, df in [("Train S1", train_s1), ("Train S2", train_s2), ("Train S3", train_s3)]:
    names_col = df["business_name"].dropna()
    counts = {}
    for pat_name, regex in suffix_patterns.items():
        counts[pat_name] = names_col.str.contains(regex, case=False, na=False).sum()
    suffix_results[name] = counts
    print(f"\n  {name}:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {k:15s}  →  {v:>10,}  ({100*v/len(names_col):.2f}%)")

# ── Chart: suffix frequency ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
patterns_sorted = sorted(suffix_patterns.keys(),
                          key=lambda k: -suffix_results["Train S1"].get(k, 0))
x_pos = np.arange(len(patterns_sorted))
w = 0.25
for i, src in enumerate(["Train S1", "Train S2", "Train S3"]):
    vals = [suffix_results[src][p] for p in patterns_sorted]
    ax.bar(x_pos + i*w, vals, w, label=src, color=PALETTE[i], edgecolor="#0d1117")
ax.set_xticks(x_pos + w)
ax.set_xticklabels(patterns_sorted, rotation=45, ha="right")
ax.set_ylabel("Occurrences")
ax.set_title("Business Name Suffix / Pattern Frequencies")
ax.legend()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1e3:.0f}K" if v >= 1000 else f"{v:.0f}"))
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "09_name_suffix_patterns.png")

# ══════════════════════════════════════════════════════════════════════════════
#  12. ADDRESS COMPONENT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
hr("12 · Address patterns — commas & components")
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for idx, (name, df) in enumerate(sources.items()):
    ax = axes[idx // 3][idx % 3]
    comma_counts = df["business_address"].dropna().str.count(",")
    ax.hist(comma_counts, bins=range(0, 15), color=PALETTE[idx], alpha=0.85, edgecolor="#0d1117")
    ax.axvline(comma_counts.median(), color="#ffa657", ls="--", lw=1.5,
               label=f"Median={comma_counts.median():.0f}")
    ax.set_title(name)
    ax.set_xlabel("Number of commas in address")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    print(f"  {name:12s}  Comma count: mean={comma_counts.mean():.1f}  "
          f"median={comma_counts.median():.0f}  max={comma_counts.max():.0f}")
fig.suptitle("Address Comma (Component) Count Distribution", fontsize=16, y=1.01, weight="bold")
fig.tight_layout()
save(fig, "10_address_comma_count.png")

# ══════════════════════════════════════════════════════════════════════════════
#  13. NON-ASCII / MULTILINGUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
hr("13 · Non-ASCII / multilingual content")
for name, df in sources.items():
    names_col = df["business_name"].dropna()
    non_ascii = names_col.apply(lambda x: bool(re.search(r"[^\x00-\x7F]", str(x))))
    n_non = non_ascii.sum()
    print(f"  {name:12s}  Non-ASCII names: {n_non:>10,}  ({100*n_non/len(names_col):.1f}%)")

# ── Chart ──
fig, ax = plt.subplots(figsize=(10, 5))
src_names = list(sources.keys())
non_ascii_pcts = []
for name, df in sources.items():
    names_col = df["business_name"].dropna()
    pct = 100 * names_col.apply(lambda x: bool(re.search(r"[^\x00-\x7F]", str(x)))).mean()
    non_ascii_pcts.append(pct)
bars = ax.bar(src_names, non_ascii_pcts, color=PALETTE[:len(src_names)], edgecolor="#0d1117")
for b, pct in zip(bars, non_ascii_pcts):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
            f"{pct:.1f}%", ha="center", va="bottom", fontsize=10, color="#c9d1d9")
ax.set_ylabel("% of records")
ax.set_title("Non-ASCII Characters in Business Names")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "11_non_ascii_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
#  14. DUPLICATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
hr("14 · Exact duplicate analysis")
for name, df in sources.items():
    dup_name = df["business_name"].dropna().duplicated(keep=False).sum()
    dup_both = df[["business_name","business_address"]].dropna().duplicated(keep=False).sum()
    print(f"  {name:12s}  Dup names: {dup_name:>10,} ({100*dup_name/len(df):.1f}%)  "
          f"Dup (name+addr): {dup_both:>10,} ({100*dup_both/len(df):.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
#  15. COUNTRY × SOURCE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
hr("15 · Country × Source cross-tab")
rows = []
for name, df in sources.items():
    for country, cnt in df["country"].value_counts().items():
        rows.append({"Source": name, "Country": country, "Count": cnt})
ctab = pd.DataFrame(rows).pivot(index="Country", columns="Source", values="Count").fillna(0).astype(int)
# reorder columns
col_order = [c for c in ["Train S1","Train S2","Train S3","Test S1","Test S2","Test S3"] if c in ctab.columns]
ctab = ctab[col_order]
print(f"\n{ctab.to_string()}\n")

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(ctab.values, aspect="auto", cmap="viridis")
ax.set_xticks(range(len(ctab.columns)))
ax.set_xticklabels(ctab.columns, rotation=30, ha="right")
ax.set_yticks(range(len(ctab.index)))
ax.set_yticklabels(ctab.index)
for i in range(len(ctab.index)):
    for j in range(len(ctab.columns)):
        val = ctab.values[i, j]
        ax.text(j, i, f"{val:,.0f}", ha="center", va="center",
                fontsize=9, color="white" if val < ctab.values.max()*0.6 else "black", weight="bold")
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Record Count", color="#c9d1d9")
cbar.ax.yaxis.set_tick_params(color="#8b949e")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")
ax.set_title("Country × Source Record Counts")
fig.tight_layout()
save(fig, "12_country_source_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
#  16. MATCH-COUNT DISTRIBUTION BY COUNTRY
# ══════════════════════════════════════════════════════════════════════════════
hr("16 · Match-count breakdown by country (train)")
gt_merged = gt.merge(train_s1[["entity_id", "country"]], left_on="source1_entity_id",
                      right_on="entity_id", how="left")

fig, ax = plt.subplots(figsize=(10, 6))
for i, country in enumerate(gt_merged["country"].dropna().unique()):
    subset = gt_merged[gt_merged["country"] == country]["num_matches"]
    print(f"  {country:>8s}  mean={subset.mean():.2f}  median={subset.median():.0f}  "
          f"max={subset.max()}  singletons={((subset==0).sum()):,}")
    ax.hist(subset, bins=range(0, 16), alpha=0.6, label=country,
            color=PALETTE[i], edgecolor="#0d1117")
ax.set_xlabel("Number of matches")
ax.set_ylabel("Frequency")
ax.set_title("Match Count Distribution by Country")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
save(fig, "13_match_count_by_country.png")

# ══════════════════════════════════════════════════════════════════════════════
#  17. SAMPLE MATCHED PAIRS (for intuition)
# ══════════════════════════════════════════════════════════════════════════════
hr("17 · Sample matched entity pairs (first 5 non-singleton)")
sample_gt = gt[gt["num_matches"] > 0].head(5)
s2_lookup = train_s2.set_index("entity_id")
s3_lookup = train_s3.set_index("entity_id")

for _, row in sample_gt.iterrows():
    s1_id = row["source1_entity_id"]
    s1_rec = train_s1[train_s1["entity_id"] == s1_id].iloc[0]
    print(f"\n  S1: {s1_id}  →  {s1_rec['business_name']}  |  {s1_rec['business_address']}  |  {s1_rec['country']}")
    for mid in row["match_list"]:
        mid = mid.strip()
        if mid.startswith("S2-") and mid in s2_lookup.index:
            r = s2_lookup.loc[mid]
            print(f"    ↳ {mid}  →  {r['business_name']}  |  {r['business_address']}  |  {r['country']}")
        elif mid.startswith("S3-") and mid in s3_lookup.index:
            r = s3_lookup.loc[mid]
            print(f"    ↳ {mid}  →  {r['business_name']}  |  {r['business_address']}  |  {r['country']}")

# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
hr("✅  EDA COMPLETE")
print(f"""
  All charts saved to:  {OUT_DIR}/
  
  Key findings to investigate further:
  ─────────────────────────────────────────────────────────────
  • Training covers US & India; test adds France (unseen country)
  • S2 & S3 are much larger than S1 (many-to-one matching)
  • Non-ASCII names (Hindi/Devanagari) are significant in S2
  • Singleton ratio and match-count distribution vary by country
  • Address formats differ across countries & sources
  ─────────────────────────────────────────────────────────────
""")
