#!/usr/bin/env python3

"""
Statistical termination-region caller for RNA Polymerase II T4ph mNET-seq data.
The method identifies termination-associated regions from fixed 50 bp binned T4ph signal profiles using gradient-based boundary detection, three-component Gaussian mixture classification and signal-based merging, adaptive distance-based merging, and final region post-processing.
Input resolution is fixed at 50 bp.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# Fixed analytical resolution of the input signal.
BIN_SIZE = 50
# Numerical safeguard preventing division by zero during MAD normalization.
EPS = 1e-8
# Internal memory-management setting. Increase for efficiency if computational resources allow it.
DEFAULT_CHUNK_SIZE = 1_000_000
# Maximum genomic distance considered for automatic estimation of the merging threshold.
MAX_MERGE_DISTANCE = 5000
# Number of histogram bins used to estimate the empirical distribution of inter-region distances during adaptive merge-distance calculation.
DISTANCE_BINS = 200
#Gaussian smoothing parameter applied to the distance distribution before local maximum detection.
DISTANCE_SMOOTHING = 8
#Maximum automatically inferred merge distance accepted as biologically meaningful.
MAX_ACCEPTED_AUTO_MERGE_DISTANCE = 1000

def parse_args():
    parser = argparse.ArgumentParser(description="Identify RNA Polymerase II transcription termination-associated regions from 50 bp binned T4ph mNET-seq signal.")
    parser.add_argument("--signal", required=True, help="Input 50 bp binned T4ph signal file: chrom, start, end, signal.")
    parser.add_argument("--gradient-z", type=float, default=3.0, help="Robust z-score threshold for detecting strong signal transitions. Higher values increase stringency. Default: 3.")
    parser.add_argument("--gmm-mode", default="background10", choices=["background10", "middle_component", "high_signal90"], help="Threshold strategy based on three-component GMM classification. background10 (default): retain regions with low posterior probability of belonging to low-signal component. middle_component: permissive threshold based on the intermediate-signal component. high_signal90: retain regions with high posterior probability of belonging to the high-signal component.")
    parser.add_argument("--merge-distance", type=int, default=None, help="Optional manual override for the distance used to merge adjacent termination-associated regions. If omitted, the distance is estimated automatically from the inter-region distance distribution.")
    parser.add_argument("--min-region-length", type=int, default=150, help="Minimum final termination-region length in bp. Shorter regions are removed as unstable predictions. Default: 150.")
    parser.add_argument("--outdir", default="statistical_caller_output", help="Output directory for predictions, summaries, and diagnostic plots.")
    return parser.parse_args()

def get_sample_name(path):
    
    """Generates output prefix from the input filename."""
   
    name = os.path.basename(path)
    name = name.rsplit(".", 1)[0]
    return name

def robust_center_scale(x):
   
    """Calculates median and median absolute deviation (MAD) for robust normalization."""
    
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return float(median), float(mad)

def iter_chromosomes(path, chunksize=DEFAULT_CHUNK_SIZE):

    """Reads a 50 bp binned signal file in chunks and yields complete chromosomes sequentially."""

    reader = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "signal"], dtype={"chrom": "string", "start": "int64", "end": "int64", "signal": "float64"}, chunksize=chunksize)
    current_chrom = None
    pieces, seen = [], set()
    for chunk in reader:
        chunk["signal"] = pd.to_numeric(chunk["signal"], errors="raise")
        if chunk["signal"].isna().any():
            raise ValueError("Missing signal values detected in input file.")
        for chrom, group in chunk.groupby("chrom", sort=False):
            chrom = str(chrom)
            if current_chrom is None:
                current_chrom = chrom
            if chrom != current_chrom:
                if chrom in seen:
                    raise ValueError("Input file must be sorted by chromosome.")
                seen.add(current_chrom)
                yield current_chrom, pd.concat(pieces, ignore_index=True)
                pieces, current_chrom = [group.copy()], chrom
            else:
                pieces.append(group.copy())
    if pieces:
        yield current_chrom, pd.concat(pieces, ignore_index=True)

def detect_signal_transitions(signal_path, gradient_z, chunksize):

    """Detects candidate gradient-based boundaries chromosome-by-chromosome. Positive gradients are normalized independently using robust median/MAD scaling, and adjacent significant bins are merged into continuous regions."""
    
    filtered_parts = []

    # Iterates through chromosomes, calculates gradients, retains only gradients which are positive
    for chrom, df in iter_chromosomes(signal_path, chunksize):
        df = df.sort_values(["start", "end"]).reset_index(drop=True)
        signal = df["signal"].to_numpy(dtype=float)
        gradient = np.gradient(signal) if len(signal) > 1 else np.array([0.0])
        positive_mask = gradient > 0
        if not np.any(positive_mask): continue
        # Calculates median and MAD per chromosome, transforms positive gradients into robust-zscores and stores only gradients exceeding chosen z-score threshold
        med, mad = robust_center_scale(gradient[positive_mask])
        selected = ((gradient - med) / (mad + EPS)) >= gradient_z
        # Merges adjacent or overlapping gradient bins into continuous regions
        if np.any(selected):
            starts, ends = df.loc[selected, "start"].to_numpy(), df.loc[selected, "end"].to_numpy()
            merged, s, e = [], starts[0], ends[0]
            for i in range(1, len(starts)):
                if starts[i] <= e:
                    e = max(e, ends[i])
                else:
                    merged.append([chrom, s, e])
                    s, e = starts[i], ends[i]
            merged.append([chrom, s, e])
            filtered_parts.append(pd.DataFrame(merged, columns=["chrom", "start", "end"]))        
    if not filtered_parts: raise RuntimeError("No significant positive gradient transitions detected.")
    return pd.concat(filtered_parts, ignore_index=True)

def define_candidate_intervals(signal_path, filtered_transitions, chunksize=DEFAULT_CHUNK_SIZE):

    """Constructs candidate intervals and calculates robust z-scores independently per chromosome."""
    
    chrom_intervals = []

    # Identify coordinates for candidate intervals
    for chrom, df in iter_chromosomes(signal_path, chunksize):
        peaks = filtered_transitions[filtered_transitions["chrom"].astype(str) == chrom]
        if len(peaks) < 2: continue
        peaks = peaks.sort_values(["start", "end"]).reset_index(drop=True)
        starts = peaks["end"].iloc[:-1].to_numpy(dtype=np.int64)
        ends = peaks["start"].iloc[1:].to_numpy(dtype=np.int64)
        valid = ends > starts
        starts, ends = starts[valid], ends[valid]
        if len(starts) == 0: continue
        bin_start, bin_end = df["start"].to_numpy(dtype=np.int64), df["end"].to_numpy(dtype=np.int64)
        sig = df["signal"].to_numpy(dtype=float)
        interval_rows = []
        # Calculates mean signal for all candidate intervals
        for gs, ge in zip(starts, ends):
            left = np.searchsorted(bin_end, gs, side="right")
            right = np.searchsorted(bin_start, ge, side="left")
            if right <= left:
                continue
            mean_signal = float(np.mean(sig[left:right]))
            interval_rows.append([chrom, int(gs), int(ge), mean_signal])
        # Calculates robust z-score per chromosome for all candidate intervals
        chrom_df = pd.DataFrame(interval_rows, columns=["chrom", "start", "end", "mean_signal"])
        med, mad = robust_center_scale(chrom_df["mean_signal"])
        chrom_df["interval_robust_z"] = (chrom_df["mean_signal"] - med) / (mad + EPS)
        chrom_intervals.append(chrom_df)    
    if not chrom_intervals: raise RuntimeError("No candidate termination intervals could be constructed.")
    return pd.concat(chrom_intervals, ignore_index=True)

def classify_candidate_intervals(intervals, gmm_mode, out_prefix):

    """
    Classifies candidate intervals using a three-component GMM.
    Components are ordered by signal intensity and interpreted as: low-signal, intermediate signal, and high-signal regions.
    Posterior probabilities are used to select termination-associated intervals according to the chosen threshold strategy.
    """

    intervals = intervals.replace([np.inf, -np.inf], np.nan)
    intervals = intervals.dropna(subset=["interval_robust_z"]).copy()
    z = intervals["interval_robust_z"].to_numpy(dtype=float)
    if len(z) < 50:
        raise RuntimeError("Insufficient data for three-component GMM fitting.")
    
    # Log-transformation of the values
    min_val = float(np.min(z))
    x = np.log1p(z - min_val).reshape(-1, 1)

    # Fitting of 3-component GMM
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42).fit(x)
    means = gmm.means_.flatten()
    order = np.argsort(means)
    m = means[order]
    s = np.sqrt(np.asarray(gmm.covariances_).reshape(-1)[order])
    w = gmm.weights_.flatten()[order]

    # Calculation of posterior probabilities
    probs = gmm.predict_proba(x)
    intervals["prob_background"] = probs[:, order[0]]
    intervals["prob_intermediate"] = probs[:, order[1]]
    intervals["prob_high_signal"] = probs[:, order[2]]

    # Application of thresholding strategy
    if gmm_mode == "background10":
        # Retains intervals with low posterior probability of belonging to the low-signal component 
        mask = intervals["prob_background"] < 0.10
    elif gmm_mode == "middle_component":
        # Retains intervals above the mean signal intensity of the intermediate-signal GMM component
        mask = x.flatten() >= m[1]
    elif gmm_mode == "high_signal90":
        # Retains only intervals with strong posterior assignment to the high-signal component
        mask = intervals["prob_high_signal"] >= 0.90
    else:
        raise ValueError(f"Unknown GMM mode: {gmm_mode}")
    gmm_supported = intervals[mask].copy()

    # Determine numeric threshold for visualization on the plot
    x_eval = np.linspace(float(x.min()), float(x.max()), 10000)
    eval_probs = gmm.predict_proba(x_eval.reshape(-1, 1))
    if gmm_mode == "background10":
        valid = np.where((x_eval >= m[0]) & (eval_probs[:, order[0]] <= 0.10))[0]
    elif gmm_mode == "middle_component":
        valid = np.where(x_eval >= m[1])[0]
    elif gmm_mode == "high_signal90":
        valid = np.where((x_eval >= m[1]) & (eval_probs[:, order[2]] >= 0.90))[0]  
    log_threshold = float(x_eval[valid[0]]) if len(valid) > 0 else m[1]
    threshold_z = float(np.expm1(log_threshold) + min_val)

    # Export 3-component GMM diagnostic figure
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(x.ravel(), bins=100, density=True, color='grey', alpha=0.4)
    labels = ["Low-signal component", "Intermediate-signal component", "High-signal component"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(3):
        ax.plot(
            x_eval, 
            norm.pdf(x_eval, m[i], s[i]) * w[i], 
            linewidth=1.5, 
            label=labels[i], 
            color=colors[i]
        )
    ax.axvline(log_threshold, linestyle="--", color="black", linewidth=1.5, label=f"Threshold ({gmm_mode})")
    ax.set_xlabel("Log-transformed robust z-score")
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_gmm_distribution.pdf", bbox_inches="tight")
    plt.close(fig)
    return gmm_supported, threshold_z

def reconstruct_supported_regions(gmm_supported, filtered_transitions):

    """Merges gradient-based boundaries based on signal."""

    combined = pd.concat([
        filtered_transitions[["chrom", "start", "end"]],
        gmm_supported[["chrom", "start", "end"]]
    ], ignore_index=True).sort_values(["chrom", "start", "end"])
    signal_merged_regions = []
    for chrom, group in combined.groupby("chrom", sort=False):
        group = group.reset_index(drop=True)
        if group.empty: 
            continue 
        start, end = int(group.loc[0, "start"]), int(group.loc[0, "end"])
        for row in group.iloc[1:].itertuples(index=False):
            if int(row.start) <= end:
                end = max(end, int(row.end))
            else:
                signal_merged_regions.append([chrom, start, end])
                start, end = int(row.start), int(row.end)        
        signal_merged_regions.append([chrom, start, end])     
    return pd.DataFrame(signal_merged_regions, columns=["chrom", "start", "end"])

def merge_supported_regions_by_distance(regions, merge_distance, out_prefix):

    """Removes unmerged 50bp supported regions and merges adjacent supported regions if the gap between them is smaller than or equal to a calculated (or user-provided) distance."""

    pre_merge_df = regions.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    pre_merge_df = pre_merge_df[(pre_merge_df["end"] - pre_merge_df["start"]) > BIN_SIZE].reset_index(drop=True)
    if merge_distance is not None:
        applied_dist = merge_distance
    else:
        distances = []
        for _, group in pre_merge_df.groupby("chrom", sort=False):
            gaps = (group["start"].shift(-1) - group["end"]).dropna()
            distances.extend(gaps[gaps > 0].tolist())   
        distances = np.asarray(distances, dtype=float)
        distances = distances[distances <= MAX_MERGE_DISTANCE]
        if len(distances) == 0:
            applied_dist = 0
        else:
            hist, edges = np.histogram(distances, bins=np.linspace(0, MAX_MERGE_DISTANCE, DISTANCE_BINS), density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            smooth = gaussian_filter1d(hist, sigma=DISTANCE_SMOOTHING)
            peaks, _ = find_peaks(smooth)
            # First local maximum is chosen
            if len(peaks) > 0:
                empirical_peak = float(centers[peaks[0]])
            else:
                # Global maximum is chosen
                max_idx = int(np.argmax(smooth))
                empirical_peak = float(centers[max_idx])
            # If peak is valid and within accepted bounds, use it. Otherwise, fallback to the 10th percentile.
            if np.isfinite(empirical_peak) and 0 < empirical_peak <= MAX_ACCEPTED_AUTO_MERGE_DISTANCE:
                applied_dist = empirical_peak
            else:
                applied_dist = np.percentile(distances, 10)     
            # Round the chosen distance to the nearest bin
            applied_dist = int(round(applied_dist / BIN_SIZE) * BIN_SIZE)

            # Export distance distribution plot
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(centers, smooth, linewidth=1.5, color='black')
            ax.axvline(applied_dist, linestyle=":", color="red", label=f"Merge distance: {applied_dist} bp")
            ax.set_xlabel("Inter-region distance (bp)")
            ax.set_ylabel("Density")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(f"{out_prefix}_distance_distribution.pdf", bbox_inches="tight")
            plt.close(fig)

    # Perform distance-based merging
    merged_rows = []
    for chrom, group in pre_merge_df.groupby("chrom", sort=False):
        group = group.reset_index(drop=True)
        if group.empty: 
            continue
        start, end = int(group.loc[0, "start"]), int(group.loc[0, "end"])
        for row in group.iloc[1:].itertuples(index=False):
            if int(row.start) - end <= applied_dist:
                end = max(end, int(row.end))
            else:
                merged_rows.append([chrom, start, end])
                start, end = int(row.start), int(row.end)
        merged_rows.append([chrom, start, end])
    return (pd.DataFrame(merged_rows, columns=["chrom", "start", "end"]), applied_dist)

def final_post_processing(merged_regions, min_length):

    """Filters short regions and shifts coordinates downstream by 1 bin to correct gradient offset."""

    df = merged_regions[merged_regions["end"] - merged_regions["start"] >= min_length].copy()
    df[["start", "end"]] += BIN_SIZE
    return df.sort_values(["chrom", "start", "end"]).reset_index(drop=True)

def write_bed(df, filepath, extra_cols=None):

    """Utility to export regions into standard BED format."""

    cols = ["chrom", "start", "end"]
    if extra_cols:
        cols.extend(extra_cols)
    df[cols].to_csv(filepath, sep="\t", header=False, index=False)

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    sample_name = get_sample_name(args.signal)
    root = os.path.join(args.outdir, sample_name)
    print(f"[{sample_name}] Detection of gradient-based boundaries.")
    filtered_transitions = detect_signal_transitions(args.signal, args.gradient_z, DEFAULT_CHUNK_SIZE)
    write_bed(filtered_transitions, f"{root}_filtered_gradients.bed")
    print(f"[{sample_name}] Construction of candidate intervals.")
    intervals = define_candidate_intervals(args.signal, filtered_transitions, DEFAULT_CHUNK_SIZE)
    print(f"[{sample_name}] Merging based on signal ({args.gmm_mode}).")
    gmm_supported, threshold = classify_candidate_intervals(intervals, args.gmm_mode, root)
    signal_merged_regions = reconstruct_supported_regions(gmm_supported, filtered_transitions)
    write_bed(signal_merged_regions, f"{root}_signal_merged.bed")
    print(f"[{sample_name}] Merging based on distance.")
    distance_merged_regions, applied_merge_distance = merge_supported_regions_by_distance(signal_merged_regions, args.merge_distance, root)
    write_bed(distance_merged_regions, f"{root}_distance_merged.bed")
    print(f"[{sample_name}] Final post-processing.")
    final_regions = final_post_processing(distance_merged_regions, args.min_region_length)
    write_bed(final_regions, f"{root}_final_termination_regions.bed")

    summary = pd.DataFrame([{
        "Sample": sample_name,
        "Signal_file": args.signal,
        "Gradient_z_threshold": args.gradient_z,
        "GMM_threshold_mode": args.gmm_mode,
        "GMM_selected_z_threshold": round(threshold, 4),
        "Detected_gradient_boundaries": len(filtered_transitions),
        "Signal_merged_regions": len(signal_merged_regions),
        "Merge_distance_bp": applied_merge_distance,
        "Distance_merged_regions": len(distance_merged_regions),
        "Min_region_length_bp": args.min_region_length,
        "Final_predicted_termination_regions": len(final_regions)
    }])
    summary.to_csv(f"{root}_summary.tsv", sep="\t", index=False)
    print(summary.T.to_string(header=False))
if __name__ == "__main__":
    main()
