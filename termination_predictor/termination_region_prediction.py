#!/usr/bin/env python3

"""
This script applies a pre-trained universal XGBoost model to predict termination-associated
regions from new 50 bp binned T4ph mNET-seq signal profiles.
"""

import argparse
import json
import os
import gc

import numpy as np
import pandas as pd
import pyranges as pr
import xgboost as xgb

# Numerical stability constant for divisions and normalization.
EPS = 1e-8
# Number of rows processed per streaming chunk during genome-wide feature extraction.
DEFAULT_CHUNKSIZE = 5_000_000


def parse_args():
    p = argparse.ArgumentParser(description="Inference pipeline for T4ph mNET-seq termination prediction.")
    p.add_argument("--signal-bins", required=True, help="50 bp binned T4ph mNET-seq signal file to predict on.")
    p.add_argument("--model", required=True, help="Path to the trained XGBoost model JSON file.")
    p.add_argument("--feature-list", required=True, help="Path to the saved feature list JSON file.")
    p.add_argument("--outdir", default="T4P_Predictions", help="Directory to save the predicted BED files.")
    p.add_argument("--out-prefix", default=None, help="Prefix for output files (defaults to input filename).")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for positive classification (default: 0.5).")
    p.add_argument("--merge-distance", type=int, default=300, help="Max distance (bp) to merge predicted bins (default: 300).")
    p.add_argument("--min-region-length", type=int, default=100, help="Minimum length (bp) for final reported regions (default: 100).")
    return p.parse_args()

def extract_name(path):

    """Generates an output prefix from the input filename."""

    return os.path.splitext(os.path.basename(path))[0]

def extract_k_values_from_features(feature_list):

    """Automatically determines the k-values used during training by parsing the feature list."""

    k_vals = set()
    for feat in feature_list:
        if feat.startswith("occ_fraction_"):
            k_vals.add(int(feat.split("_")[-1]))
    if not k_vals:
        raise ValueError("Could not auto-detect k-values from the feature list. Ensure the feature list is correct.")
    return sorted(list(k_vals))

def robust_normalize_signal(df, signal_col="signal", norm_col="norm_signal"):

    """Performs robust normalization of signal intensity using median and median absolute deviation (MAD)."""

    x = df[signal_col].to_numpy(dtype=float)
    nz = x[x > 0]
    if len(nz) > 0:
        med = np.median(nz)
        mad = np.median(np.abs(nz - med))
    else:
        med, mad = 0.0, 1.0
    norm_x = np.where(x > 0, (x - med) / (mad + EPS), 0.0)
    df[norm_col] = norm_x.astype("float32")
    return df, med, mad

# Feature extraction

def add_local_features(df, norm_col="norm_signal"):

    """Calculates local signal-based features from normalized T4ph mNET-seq profiles."""

    x = df[norm_col].astype("float32")
    features = pd.DataFrame(index=df.index)
    features["norm_signal_local"] = x
    features["rolling_std_5"] = x.rolling(5, center=True, min_periods=1).std().fillna(0.0).astype("float32")
    return features

def add_window_features(df, k_values, norm_col="norm_signal"):

    """Calculates multi-scale window-based features from normalized signal."""

    x = df[norm_col].astype("float32")
    features = pd.DataFrame(index=df.index)
    for k in k_values:
        window = 2 * k + 1
        roll = x.rolling(window, center=True, min_periods=1)
        w_max = roll.max().fillna(0.0)
        features[f"win{k}_fold_change"] = (w_max / (float(w_max.mean()) + EPS)).astype("float32")
        features[f"win{k}_max"] = w_max.astype("float32")
        features[f"win{k}_std"] = roll.std().fillna(0.0).astype("float32")
        features[f"win{k}_mean"] = roll.mean().astype("float32")
    return features

def add_occupancy_features(df, k_values):

    """Calculates multi-scale occupancy-based features from signal."""

    features = pd.DataFrame(index=df.index)
    occupied = (df["signal"] > 0).astype("float32")
    sig_vals = df["signal"].astype("float32")
    for k in k_values:
        window = 2 * k + 1
        occ = occupied.rolling(window, center=True, min_periods=1)
        features[f"occ_fraction_{k}"] = occ.mean().astype("float32")
        left_sig = sig_vals.shift(1).rolling(k, min_periods=1).sum().fillna(0)
        right_sig = sig_vals.iloc[::-1].rolling(k, min_periods=1).sum().iloc[::-1].shift(-1).fillna(0)
        total_lr_sig = left_sig + right_sig + EPS
        features[f"signal_asymmetry_{k}"] = ((right_sig - left_sig) / total_lr_sig).astype("float32")
    return features

def build_features(df, k_values):

    """Constructs the complete feature matrix used for ML prediction."""

    df = df.sort_values(["start", "end"]).reset_index(drop=True)
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0.0).astype("float32")
    df, _, _ = robust_normalize_signal(df, signal_col="signal", norm_col="norm_signal")
    
    f1 = add_local_features(df, norm_col="norm_signal")
    f2 = add_window_features(df, k_values, norm_col="norm_signal")
    f3 = add_occupancy_features(df, k_values)
    
    feature_df = pd.concat([f1, f2, f3], axis=1)
    df = pd.concat([df, feature_df], axis=1)
    return df

def compute_global_stats(bin_path, chunksize):

    """Computes global signal occupancy statistics using memory-efficient streaming."""

    total_bins, total_nz, total_isolated = 0, 0, 0
    reader = pd.read_csv(bin_path, sep="\t", header=None, usecols=[0, 3], names=["chrom", "signal"], dtype={"chrom": "string", "signal": "float32"}, chunksize=chunksize)
    for chunk in reader:
        total_bins += len(chunk)
        for chrom, group in chunk.groupby("chrom", sort=False):
            nz = group["signal"].to_numpy() > 0
            total_nz += int(nz.sum())

            left = np.zeros(len(nz), dtype=bool)
            right = np.zeros(len(nz), dtype=bool)
            if len(nz) > 1:
                left[1:] = nz[:-1]
                right[:-1] = nz[1:]
            total_isolated += int((nz & ~left & ~right).sum())
            
    occupancy = total_nz / max(1, total_bins)
    isolated_frac = total_isolated / max(1, total_nz)
    return occupancy, isolated_frac

def iter_chromosomes(bin_path, chunksize):

    """Iterates through a genome-wide binned signal file and yields complete chromosomes sequentially."""

    current_chrom = None
    pieces = []
    reader = pd.read_csv(bin_path, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["chrom", "start", "end", "signal"], dtype={"chrom": "string", "start": "int32", "end": "int32", "signal": "float32"}, chunksize=chunksize)
    for chunk in reader:
        if chunk.empty:
            continue
        for chrom, group in chunk.groupby("chrom", sort=False):
            if current_chrom is None:
                current_chrom = chrom
            if chrom != current_chrom:
                yield current_chrom, pd.concat(pieces, ignore_index=True)
                pieces = [group]
                current_chrom = chrom
            else:
                pieces.append(group)
    if pieces:
        yield current_chrom, pd.concat(pieces, ignore_index=True)

# Prediction and post-processing 

def bins_to_adjacent_regions(pred_bins):

    """Merges overlapping or directly touching predicted bins into continuous genomic regions."""

    if pred_bins.empty: return pd.DataFrame(columns=["chrom", "start", "end", "max_prob"])
    rows = []
    for chrom, group in pred_bins.groupby("chrom", sort=False):
        group = group.sort_values(["start", "end"]).reset_index(drop=True)
        c_start, c_end, probs = int(group.loc[0, "start"]), int(group.loc[0, "end"]), [float(group.loc[0, "xgb_prob"])]
        for row in group.iloc[1:].itertuples(index=False):
            if int(row.start) <= c_end:
                c_end = max(c_end, int(row.end))
                probs.append(float(row.xgb_prob))
            else:
                rows.append([chrom, c_start, c_end, max(probs)])
                c_start, c_end, probs = int(row.start), int(row.end), [float(row.xgb_prob)]
        rows.append([chrom, c_start, c_end, max(probs)])
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "max_prob"])

def merge_regions_only(raw_regions, merge_distance):

    """Merges termination-associated regions separated by a gap up to `merge_distance`."""

    if raw_regions.empty: return raw_regions.copy()
    rows = []
    for chrom, group in raw_regions.groupby("chrom", sort=False):
        group = group.sort_values(["start", "end"]).reset_index(drop=True)
        c_start, c_end = int(group.loc[0, "start"]), int(group.loc[0, "end"])
        max_probs = [float(group.loc[0, "max_prob"])]
        for row in group.iloc[1:].itertuples(index=False):
            if int(row.start) - c_end <= merge_distance:
                c_end = max(c_end, int(row.end))
                max_probs.append(float(row.max_prob))
            else:
                rows.append([chrom, c_start, c_end, max(max_probs)])
                c_start, c_end, max_probs = int(row.start), int(row.end), [float(row.max_prob)]
        rows.append([chrom, c_start, c_end, max(max_probs)])
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "max_prob"])

def apply_boundary_correction(merged_regions, pos_bins):

    """Corrects predicted region boundaries by restricting them strictly to non-zero signal bins."""

    if merged_regions.empty: return merged_regions.copy()
    if pos_bins.empty: return pd.DataFrame(columns=["chrom", "start", "end", "max_prob"])
    
    merged_regions = merged_regions.copy()
    merged_regions["region_id"] = np.arange(len(merged_regions))
    pr_regions = pr.PyRanges(merged_regions.rename(columns={"chrom": "Chromosome", "start": "Start", "end": "End"}))
    pr_bins = pr.PyRanges(pos_bins.rename(columns={"chrom": "Chromosome", "start": "Start", "end": "End"}))
    
    joined = pr_regions.join(pr_bins).as_df()
    if joined.empty: return pd.DataFrame(columns=["chrom", "start", "end", "max_prob"])
    
    rows = []
    for _, group in joined.groupby("region_id"):
        chrom = group["Chromosome"].iloc[0]
        max_prob = group["max_prob"].iloc[0]
        rows.append([chrom, group["Start_b"].min(), group["End_b"].max(), max_prob])
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "max_prob"])

def filter_by_length(regions, min_region_length):

    """Filters out predicted termination regions shorter than the minimum length threshold."""

    if regions.empty: return regions.copy()
    filtered = regions[(regions["end"] - regions["start"]) >= min_region_length].copy()
    return filtered.reset_index(drop=True)

# Main execution

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    
    out_prefix = args.out_prefix if args.out_prefix else extract_name(args.signal_bins)
    base_out_path = os.path.join(args.outdir, out_prefix)
    model = xgb.XGBClassifier()
    model.load_model(args.model)
    with open(args.feature_list, "r") as f:
        sig_feats = json.load(f)

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != sig_feats:
            raise ValueError("Feature list does not match the features used during model training.")

    k_vals = extract_k_values_from_features(sig_feats)
    print(f"Auto-detected multi-scale window parameters (k-values): {k_vals}")
    print("\n Computing of global occupancy statistics")
    occupancy, isolated_frac = compute_global_stats(args.signal_bins, DEFAULT_CHUNKSIZE)
    print("\n Chromosome-wise feature extraction and prediction")
    all_called_bins = []
    all_pos_bins = []
    for chrom, chrom_df in iter_chromosomes(args.signal_bins, DEFAULT_CHUNKSIZE):
        chrom_pos_bins = chrom_df[chrom_df["signal"] > 0][["chrom", "start", "end"]].copy()
        all_pos_bins.append(chrom_pos_bins)
        cdf = build_features(chrom_df, k_vals)
        if "sample_signal_occupancy" in sig_feats:
            cdf["sample_signal_occupancy"] = np.float32(occupancy)

        if "sample_isolated_fraction" in sig_feats:
            cdf["sample_isolated_fraction"] = np.float32(isolated_frac)

        missing = set(sig_feats) - set(cdf.columns)
        if missing:
            raise ValueError(f"Extracted data is missing features expected by the model: {missing}")
        X_infer = cdf.loc[:, sig_feats]
        probs = model.predict_proba(X_infer)[:, 1].astype("float32")
        cdf["xgb_prob"] = probs
        called = cdf[cdf["xgb_prob"] >= args.threshold][["chrom", "start", "end", "xgb_prob"]].copy()
        all_called_bins.append(called)
        del chrom_df, chrom_pos_bins, cdf, X_infer, probs, called
        gc.collect()

    print("\n Post-processing")
    
    final_called_bins = pd.concat(all_called_bins, ignore_index=True) if all_called_bins else pd.DataFrame()
    final_pos_bins = pd.concat(all_pos_bins, ignore_index=True) if all_pos_bins else pd.DataFrame()    
    raw_regions = bins_to_adjacent_regions(final_called_bins)
    length_filtered_regions = filter_by_length(raw_regions, args.min_region_length)
    merged_regions = merge_regions_only(length_filtered_regions, args.merge_distance)    
    corrected_regions = apply_boundary_correction(merged_regions, final_pos_bins)
    final_regions = filter_by_length(corrected_regions, args.min_region_length)

    final_regions = final_regions.sort_values(
        ["chrom", "start", "end"]
    ).reset_index(drop=True)

    out_file = f"{base_out_path}_final_termination_regions.bed"

    if not final_regions.empty:
        final_regions.to_csv(
            out_file,
            sep="\t",
            header=False,
            index=False
        )
    else:
        open(out_file, 'w').close()     
if __name__ == "__main__":
    main()
