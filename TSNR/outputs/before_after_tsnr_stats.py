import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

# ============================================================
# Whole-brain tSNR before vs after NORDIC
# ------------------------------------------------------------
# What this script does:
# 1) Loads full 4D fMRI runs before and after NORDIC
# 2) Computes voxelwise tSNR = temporal mean / temporal std
# 3) Keeps only valid whole-brain voxels (finite, std>0, mean>0)
# 4) Summarizes run-level whole-brain tSNR
# 5) Tests whether after > before across runs
# 6) Creates a publication-style figure
# ============================================================


# =========================
# CONFIG: EDIT THESE PATHS
# =========================
RUNS = [1, 2, 3]

OUTPUT_ROOT = r"D:/NORDIC_Subject1_project/TSNR"

def before_nordic_path(run: int) -> str:
    before_map = {
        1: r"D:/Downloads/run1/func/mag_POCS_r1.nii.gz",
        2: r"D:/Downloads/run2/func/mag_POCS_r2.nii.gz",
        3: r"D:/Downloads/run3/func/mag_POCS_r3.nii.gz",
    }
    return before_map[run]

def after_nordic_path(run: int) -> str:
    after_map = {
        1: r"D:/NORDIC_Subject1_project/GLM after nordic/NORDIC_run1_mag_fixhdr.nii.gz",
        2: r"D:/NORDIC_Subject1_project/GLM after nordic/NORDIC_run2_mag_fixhdr.nii.gz",
        3: r"D:/NORDIC_Subject1_project/GLM after nordic/NORDIC_run3_mag_fixhdr.nii.gz",
    }
    return after_map[run]


# =========================
# IO
# =========================
def load_4d(path):
    """
    Load a NIfTI file as float32 4D data.

    Why nibabel?
    nibabel is the standard Python library for reading NIfTI neuroimaging files.

    What this function checks:
    - file exists
    - image is 4D
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D data, got shape {data.shape}: {path}")
    return data, img


# =========================
# tSNR core
# =========================
def voxelwise_tsnr(data_4d):
    """
    Compute voxelwise tSNR from a 4D run.

    Formula:
    tSNR(v) = mean_t(v) / std_t(v)

    mean_t(v):
        temporal mean of one voxel across all timepoints
    std_t(v):
        temporal standard deviation of the same voxel

    Returns:
    - tsnr_map: 3D voxelwise tSNR
    - valid_mask: 3D boolean mask for valid whole-brain voxels

    Why valid_mask is needed:
    We exclude voxels that are non-finite, zero-variance, or zero-signal,
    because they would produce meaningless tSNR values.
    """
    mean_vol = np.mean(data_4d, axis=3)
    std_vol = np.std(data_4d, axis=3)

    valid_mask = (
        np.isfinite(mean_vol) &
        np.isfinite(std_vol) &
        (std_vol > 0) &
        (mean_vol > 0)
    )

    tsnr_map = np.zeros(mean_vol.shape, dtype=np.float32)
    tsnr_map[valid_mask] = mean_vol[valid_mask] / std_vol[valid_mask]

    return tsnr_map, valid_mask


def summarize_tsnr(tsnr_map, valid_mask):
    """
    Summarize whole-brain tSNR from all valid voxels.

    Why mean and median?
    - Mean gives an overall global level.
    - Median is more robust to outliers.
    """
    vals = tsnr_map[valid_mask]
    if vals.size == 0:
        raise ValueError("No valid voxels found for tSNR calculation.")

    return {
        "n_valid_voxels": int(vals.size),
        "mean_tsnr": float(np.mean(vals)),
        "median_tsnr": float(np.median(vals)),
        "std_tsnr": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    }


# =========================
# Plotting
# =========================
def make_figure(df, out_path):
    """
    Create a clean two-panel figure.

    Panel A:
    paired slope plot for before vs after NORDIC

    Panel B:
    percent change per run

    Why this figure?
    - The slope plot shows within-run direction of change clearly.
    - The percent-change bars show effect size intuitively.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # ---------- Panel A: paired slope plot ----------
    ax = axes[0]
    x = [0, 1]

    before_vals = df["before_mean_tsnr"].values
    after_vals = df["after_mean_tsnr"].values

    for i, row in df.iterrows():
        color = "#1f77b4" if row["delta_mean_tsnr"] >= 0 else "#d62728"
        ax.plot(
            x,
            [row["before_mean_tsnr"], row["after_mean_tsnr"]],
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=color,
            alpha=0.9
        )
        ax.text(1.03, row["after_mean_tsnr"], f"Run {int(row['run'])}", fontsize=9, va="center")

    # group mean line
    ax.plot(
        x,
        [before_vals.mean(), after_vals.mean()],
        color="black",
        linewidth=4,
        marker="s",
        markersize=10,
        label="Mean across runs"
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before NORDIC", "After NORDIC"])
    ax.set_ylabel("Mean whole-brain tSNR")
    ax.set_title("Whole-brain tSNR change")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    # equation box
    eq_text = (
        "tSNR(v) = mean_t(v) / std_t(v)\n"
        "mean_t = temporal mean\n"
        "std_t = temporal SD"
    )
    ax.text(
        0.03, 0.05, eq_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="0.5")
    )

    # ---------- Panel B: percent-change bars ----------
    ax2 = axes[1]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in df["percent_change"]]
    bars = ax2.bar(df["run"].astype(str), df["percent_change"], color=colors, alpha=0.9)

    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("Run")
    ax2.set_ylabel("% change")
    ax2.set_title("Percent change after NORDIC")
    ax2.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, df["percent_change"]):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + (0.8 if val >= 0 else -0.8),
            f"{val:.1f}%",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# =========================
# Main analysis
# =========================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    rows = []

    for run in RUNS:
        before_path = before_nordic_path(run)
        after_path = after_nordic_path(run)

        print(f"\nRun {run}: loading BEFORE...")
        before_data, before_img = load_4d(before_path)

        print(f"Run {run}: loading AFTER...")
        after_data, after_img = load_4d(after_path)

        print(f"Run {run}: computing voxelwise tSNR...")
        before_tsnr_map, before_valid = voxelwise_tsnr(before_data)
        after_tsnr_map, after_valid = voxelwise_tsnr(after_data)

        # Whole-brain summary for each image independently
        before_summary = summarize_tsnr(before_tsnr_map, before_valid)
        after_summary = summarize_tsnr(after_tsnr_map, after_valid)

        rows.append({
            "run": run,
            "before_n_valid_voxels": before_summary["n_valid_voxels"],
            "after_n_valid_voxels": after_summary["n_valid_voxels"],
            "before_mean_tsnr": before_summary["mean_tsnr"],
            "after_mean_tsnr": after_summary["mean_tsnr"],
            "delta_mean_tsnr": after_summary["mean_tsnr"] - before_summary["mean_tsnr"],
            "before_median_tsnr": before_summary["median_tsnr"],
            "after_median_tsnr": after_summary["median_tsnr"],
            "delta_median_tsnr": after_summary["median_tsnr"] - before_summary["median_tsnr"],
            "percent_change": (
                (after_summary["mean_tsnr"] - before_summary["mean_tsnr"])
                / before_summary["mean_tsnr"] * 100.0
            )
        })

    # -------------------------
    # Per-run table
    # -------------------------
    df = pd.DataFrame(rows)
    table_csv = os.path.join(OUTPUT_ROOT, "wholebrain_tsnr_before_after_table.csv")
    df.to_csv(table_csv, index=False)

    # -------------------------
    # Statistics across runs
    # -------------------------
    before = df["before_mean_tsnr"].values
    after = df["after_mean_tsnr"].values
    diff = after - before

    # Paired t-test:
    # tests whether mean difference across runs differs from zero
    t_stat, t_p = ttest_rel(after, before)

    # Wilcoxon:
    # non-parametric paired alternative for small samples
    try:
        w_stat, w_p = wilcoxon(after, before)
    except Exception:
        w_stat, w_p = np.nan, np.nan

    stats_df = pd.DataFrame([{
        "n_runs": len(df),
        "mean_before": float(np.mean(before)),
        "sd_before": float(np.std(before, ddof=1)) if len(before) > 1 else 0.0,
        "mean_after": float(np.mean(after)),
        "sd_after": float(np.std(after, ddof=1)) if len(after) > 1 else 0.0,
        "mean_difference": float(np.mean(diff)),
        "sd_difference": float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0,
        "mean_percent_change": float(np.mean(df["percent_change"])),
        "paired_t_stat": float(t_stat),
        "paired_t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat) if np.isfinite(w_stat) else np.nan,
        "wilcoxon_pvalue": float(w_p) if np.isfinite(w_p) else np.nan
    }])

    stats_csv = os.path.join(OUTPUT_ROOT, "wholebrain_tsnr_before_after_stats.csv")
    stats_df.to_csv(stats_csv, index=False)

    # -------------------------
    # Text report
    # -------------------------
    report_path = os.path.join(OUTPUT_ROOT, "wholebrain_tsnr_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Whole-brain tSNR before vs after NORDIC\n")
        f.write("======================================\n\n")
        f.write("Definition:\n")
        f.write("tSNR(v) = mean_t(v) / std_t(v)\n\n")
        f.write("This analysis was performed on whole-brain valid voxels from each 4D run.\n")
        f.write("A voxel was considered valid if mean>0, std>0, and both values were finite.\n\n")

        f.write("Per-run results:\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

        f.write("Group summary across runs:\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n")

        f.write("Interpretation notes:\n")
        f.write("- Higher tSNR means better temporal stability relative to temporal noise.\n")
        f.write("- Positive delta_mean_tsnr means NORDIC improved whole-brain tSNR.\n")
        f.write("- With only 3 runs, formal p-values should be interpreted cautiously.\n")

    # -------------------------
    # Figure
    # -------------------------
    fig_path = os.path.join(OUTPUT_ROOT, "wholebrain_tsnr_before_after_figure.png")
    make_figure(df, fig_path)

    print("\nDone.")
    print("Saved:")
    print(table_csv)
    print(stats_csv)
    print(report_path)
    print(fig_path)


if __name__ == "__main__":
    main()