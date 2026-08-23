#!/usr/bin/env python3

"""
Machine learning pipeline for predicting RNA Polymerase II transcription termination
regions from T4ph mNET-seq signal profiles.
The pipeline performs chromosome-wise feature extraction from 50 bp binned signal data,
trains a universal XGBoost classifier using experimentally defined termination regions,
and evaluates model generalization using leave-one-cell-line-out validation.
Outputs include the trained model, feature definitions, validation metrics, and
predicted termination-associated regions.
"""

import argparse
import gc
import json
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyranges as pr
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

# Numerical stability constant for divisions and normalization.
EPS = 1e-8
# Global random seed.
SEED = 42
# Maximum number of CPU threads used by parallelized libraries.
N_JOBS = 32
# Number of rows processed per streaming chunk during genome-wide feature extraction.
DEFAULT_CHUNKSIZE = 5_000_000


def parse_args():
    p = argparse.ArgumentParser(description="Universal XGBoost classifier for T4ph mNET-seq termination prediction.")
    p.add_argument("--signal-bins", nargs="+", required=True, help="50 bp binned T4ph mNET-seq signal files.")
    p.add_argument("--positive-regions", nargs="+", required=True, help="BED files containing positive training regions.")
    p.add_argument("--data-dir", default="T4P_Universal_Results", help="Output directory for all pipeline results.")
    p.add_argument("--k-list", default="3,5,10,15,30,50,80", help="Window sizes (bins) for multi-scale feature extraction.")
    p.add_argument("--neg-ratio", type=int, default=10, help="Ratio of negative background regions per positive region.")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for positive classification.")
    p.add_argument("--merge-distance", type=int, default=300, help="Max distance (bp) to merge predicted bins.")
    p.add_argument("--min-region-length", type=int, default=150, help="Minimum length (bp) for final reported regions.")
    p.add_argument("--n-estimators", type=int, default=300, help="Number of boosting trees.")
    p.add_argument("--max-depth", type=int, default=3, help="Maximum depth of decision trees.")
    p.add_argument("--learning-rate", type=float, default=0.05, help="Boosting learning rate.")
    return p.parse_args()


def extract_name(path):

    """Generates output prefix from the input filename."""

    return os.path.splitext(os.path.basename(path))[0]


def parse_int_list(text):

    """Converts a comma-separated string into a sorted list of unique integers. Used for parsing command-line parameters requiring multiple integer values."""

    return sorted({int(x.strip()) for x in text.split(",") if x.strip()})


def robust_normalize_signal(df, signal_col="signal", norm_col="norm_signal"):

    """
    Performs robust normalization of signal intensity using median and median absolute deviation (MAD) scaling.
    Scaling parameters are calculated only from non-zero signal bins.
    """

    x = df[signal_col].to_numpy(dtype=float)
    nz = x[x > 0]
    if len(nz) > 0:
        med = np.median(nz)
        mad = np.median(np.abs(nz - med))
    else:
        med, mad = 0.0, 1.0
    norm_x = np.where(
        x > 0,
        (x - med) / (mad + EPS),
        0.0
    )
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

    """
    Calculates multi-scale window-based features from normalized T4ph mNET-seq signal.
    For each genomic bin, features are calculated over symmetric windows of
    different sizes around the bin. Window size is defined as (2*k + 1) bins, where each bin corresponds to the input signal resolution.
    """

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

    """Calculates multi-scale occupancy-based features from T4ph mNET-seq signal."""

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

    """Constructs the complete feature matrix used for machine learning prediction of RNA Polymerase II termination-associated regions."""

    df = df.sort_values(["start", "end"]).reset_index(drop=True)
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").fillna(0.0).astype("float32")
    df, _, _ = robust_normalize_signal(df, signal_col="signal", norm_col="norm_signal")
    f1 = add_local_features(df, norm_col="norm_signal")
    f2 = add_window_features(df, k_values, norm_col="norm_signal")
    f3 = add_occupancy_features(df, k_values)
    feature_df = pd.concat([f1, f2, f3], axis=1)
    feature_names = list(feature_df.columns)
    df = pd.concat([df, feature_df], axis=1)
    return df, feature_names

def label_bins(df, positive_df):

    """Assigns binary training labels to genomic bins based on overlap with statistically defined termination-associated regions."""

    labels = np.zeros(len(df), dtype=np.int8)
    if positive_df.empty: return labels

    bins_pr = pd.DataFrame({
        "Chromosome": df["chrom"].astype(str),
        "Start": df["start"].to_numpy(),
        "End": df["end"].to_numpy(),
        "_row_id": np.arange(len(df), dtype=np.int64)
    })
    pos_pr = pd.DataFrame({
        "Chromosome": positive_df["chrom"].astype(str),
        "Start": positive_df["start"].to_numpy(),
        "End": positive_df["end"].to_numpy()
    })

    counted = pr.PyRanges(bins_pr).count_overlaps(pr.PyRanges(pos_pr)).as_df()
    overlaps = counted[counted["NumberOverlaps"] > 0]["_row_id"].to_numpy()
    labels[overlaps] = 1
    return labels

def compute_global_stats(bin_path, chunksize):

    """Computes global signal occupancy statistics from a genome-wide binned T4ph mNET-seq signal file using memory-efficient streaming."""

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

    """Iterates through a genome-wide binned T4ph mNET-seq signal file and yields complete chromosomes sequentially."""

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

def process_sample(bin_path, pos_path, k_values_str, chunksize, args, rng):

    """Processes a single T4ph mNET-seq sample for machine learning model training."""

    k_values = parse_int_list(k_values_str)
    sample_id = extract_name(bin_path)

    print(f"  [{sample_id}] Calculation of global cell-specific features.")
    occupancy, isolated_frac = compute_global_stats(bin_path, chunksize)
    truth = pd.read_csv(pos_path, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], dtype={"chrom": "string", "start": "int32", "end": "int32"})
    truth = truth.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    out_parquet = os.path.join(args.data_dir, f"{sample_id}_features_cache.parquet")
    writer = None
    sampled_pieces, final_sig_feats = [], []

    print(f"  [{sample_id}] Chromosome-wise feature generation.")
    for chrom, chrom_df in iter_chromosomes(bin_path, chunksize):
        chrom_truth = truth[truth["chrom"] == chrom]
        chrom_df["label"] = label_bins(chrom_df, chrom_truth)
        cdf, feats = build_features(chrom_df, k_values)

        cdf["sample_signal_occupancy"] = np.float32(occupancy)
        cdf["sample_isolated_fraction"] = np.float32(isolated_frac)
        cdf["sample_id"] = sample_id
        final_sig_feats = feats + ["sample_signal_occupancy", "sample_isolated_fraction"]

        cols_for_parquet = ["chrom", "start", "end", "signal"] + final_sig_feats + ["label"]
        table = pa.Table.from_pandas(cdf[cols_for_parquet])

        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(out_parquet, schema)
        writer.write_table(table)

        pos_idx = cdf.index[cdf["label"] == 1].to_numpy()
        neg_idx = cdf.index[cdf["label"] == 0].to_numpy()
        n_neg = min(len(neg_idx), len(pos_idx) * args.neg_ratio)

        if len(pos_idx) > 0 or n_neg > 0:
            selected = np.concatenate([pos_idx, rng.choice(neg_idx, size=n_neg, replace=False)]) if n_neg > 0 else pos_idx.copy()
            cols_to_keep = final_sig_feats + ["label", "sample_id"]
            sampled_pieces.append(cdf.loc[selected, cols_to_keep].copy())

        del chrom_df, cdf, table, chrom_truth; gc.collect()

    if writer is not None: writer.close()
    train_sample_df = pd.concat(sampled_pieces, ignore_index=True) if sampled_pieces else pd.DataFrame()
    del truth, sampled_pieces; gc.collect()
    return train_sample_df, final_sig_feats, out_parquet

# Training and prediction

def train_universal_model(train_dfs, feats, args):

    """Trains a universal XGBoost classifier combining data from all provided samples."""

    pieces = []
    for df in train_dfs:
        if df.empty: continue
        # Feature consistency check
        missing_feats = set(feats) - set(df.columns)
        if missing_feats:
            sample_name = df["sample_id"].iloc[0] if "sample_id" in df.columns else "Unknown"
            raise ValueError(f"Sample {sample_name} is missing expected features: {missing_feats}")

        # Sample-level weights to prevent samples with larger numbers of training bins from dominating model optimization
        part = df.copy()
        part["_sample_weight"] = np.float32(1.0 / len(part))
        pieces.append(part)
        print(f"    {part['sample_id'].iloc[0]}: Total Train Samples (pos+neg) = {len(part):,}")

    if not pieces: return None
    train = pd.concat(pieces, ignore_index=True)

    if train["label"].nunique() < 2:
        raise ValueError("Training data must contain both positive and negative examples.")

    weights = train["_sample_weight"].to_numpy() * (len(train) / train["_sample_weight"].sum())

    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=SEED,
        n_jobs=N_JOBS,
        tree_method="hist",
        eval_metric="logloss",
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0
    )
    model.fit(
        train[feats],
        train["label"],
        sample_weight=weights
    )
    return model

def predict_universal(test_df, model, feats, chunksize=DEFAULT_CHUNKSIZE):

    """Generates XGBoost probability predictions in batches to reduce peak memory usage during model inference."""

    if model is None:
        raise ValueError("A trained model is required for prediction.")

    missing_feats = set(feats) - set(test_df.columns)
    if missing_feats:
        raise ValueError(f"Test data is missing expected features: {missing_feats}")

    probs = np.zeros(len(test_df), dtype=np.float32)
    for i in range(0, len(test_df), chunksize):
        chunk = test_df[feats].iloc[i:i+chunksize]
        probs[i:i+chunksize] = model.predict_proba(chunk)[:, 1].astype("float32")

    test_df["xgb_prob"] = probs
    return test_df

# Diagnostic plots and post-processing of the regions

def generate_diagnostic_plots(test_df, model, threshold, out_prefix):

    """Generates and saves model performance diagnostics (ROC, PR, confusion matrix, feature importance)."""

    y_true, y_prob = test_df["label"].to_numpy(), test_df["xgb_prob"].to_numpy()
    y_pred = (y_prob >= threshold).astype(int)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)
    ax_roc, ax_pr = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_cm, ax_fi = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    if np.unique(y_true).size == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_score(y_true, y_prob):.3f}')
        ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax_roc.set_title('ROC curve')
        ax_roc.legend(loc="lower right")
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax_pr.plot(recall, precision, color='purple', lw=2, label=f'PR-AUC = {average_precision_score(y_true, y_prob):.3f}')
        ax_pr.set_title('Precision-recall curve')
        ax_pr.legend(loc="lower left")

    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax_cm)
    ax_cm.set_title(f'Confusion matrix (cutoff = {threshold:.2f})')

    if model:
        gain = sorted(model.get_booster().get_score(importance_type="gain").items(), key=lambda x: x[1], reverse=True)[:10]
        if gain:
            x_vals = [v for k, v in gain]
            y_vals = [f"(S) {k}" for k, v in gain]
            sns.barplot(x=x_vals, y=y_vals, hue=y_vals, ax=ax_fi, palette="viridis", legend=False)
            ax_fi.set_title('Top 10 features')

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_visual_diagnostics.png", dpi=300)
    plt.close()

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

def apply_boundary_correction(merged_regions, test_df):

    """Corrects predicted region boundaries by restricting them strictly to non-zero signal bins."""

    if merged_regions.empty: return merged_regions.copy()
    pos_bins = test_df[test_df["signal"] > 0][["chrom", "start", "end"]]
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
    rng = np.random.default_rng(SEED)
    os.makedirs(args.data_dir, exist_ok=True)
    eval_thresholds = [args.threshold]
    train_data = {}
    parquet_paths = {}
    sig_feats = []

    print("\n Feature generation")
    for b_path, p_path in zip(args.signal_bins, args.positive_regions):
        sample_id = extract_name(b_path)
        train_sample, feats, out_parquet = process_sample(b_path, p_path, args.k_list, DEFAULT_CHUNKSIZE, args, rng)
        train_data[sample_id] = train_sample
        parquet_paths[sample_id] = out_parquet
        sig_feats = feats

    samples = list(train_data.keys())
    metrics_log = []

    print("\n Universal LOCO validation")
    for fold, held_out in enumerate(samples, 1):
        print(f"\n Fold {fold}/{len(samples)}: Test sample = {held_out}")
        all_train_dfs = [train_data[s] for s in samples if s != held_out]
        model = train_universal_model(all_train_dfs, sig_feats, args)

        if model:
            booster = model.get_booster()
            gain_scores = booster.get_score(importance_type="gain")
            weight_scores = booster.get_score(importance_type="weight")
            imp_df = pd.DataFrame({
                "feature": list(gain_scores.keys()),
                "gain": list(gain_scores.values()),
                "weight": [weight_scores.get(f, 0) for f in gain_scores.keys()]
            }).sort_values(by="gain", ascending=False)
            imp_df.to_csv(os.path.join(args.data_dir, f"{held_out}_feature_importance.csv"), index=False)

        test_df = pd.read_parquet(parquet_paths[held_out])
        test_df = predict_universal(test_df, model, sig_feats, chunksize=DEFAULT_CHUNKSIZE)
        y_true = test_df["label"].to_numpy()
        y_prob = test_df["xgb_prob"].to_numpy()

        pr_auc = float(average_precision_score(y_true, y_prob)) if np.unique(y_true).size == 2 else np.nan
        roc_auc = float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size == 2 else np.nan
        print(f"Global PR-AUC: {pr_auc:.4f} ROC-AUC: {roc_auc:.4f}")

        for thresh in eval_thresholds:
            print(f"Enforcing threshold: {thresh}")
            y_pred_raw = (y_prob >= thresh).astype(np.int8)
            generate_diagnostic_plots(test_df, model, thresh, os.path.join(args.data_dir, f"{held_out}_t{thresh}"))

            called_bins = test_df.loc[y_pred_raw == 1, ["chrom", "start", "end", "xgb_prob"]]
            raw_regions = bins_to_adjacent_regions(called_bins)
            y_pred_stage_raw = y_pred_raw

            length_filtered_regions = filter_by_length(raw_regions, args.min_region_length)

            merged_regions = merge_regions_only(length_filtered_regions, args.merge_distance)
            y_pred_stage_merge = label_bins(test_df, merged_regions)

            corrected_regions = apply_boundary_correction(merged_regions, test_df)
            filtered_regions = filter_by_length(corrected_regions, args.min_region_length)

            y_pred_stage_correct = label_bins(test_df, corrected_regions)
            y_pred_stage_filter = label_bins(test_df, filtered_regions)

            def calc_metrics(y_t, y_p):
                return (
                    float(precision_score(y_t, y_p, zero_division=0)),
                    float(recall_score(y_t, y_p, zero_division=0)),
                    float(f1_score(y_t, y_p, zero_division=0))
                )

            r_prec, r_rec, r_f1 = calc_metrics(y_true, y_pred_stage_raw)
            m_prec, m_rec, m_f1 = calc_metrics(y_true, y_pred_stage_merge)
            c_prec, c_rec, c_f1 = calc_metrics(y_true, y_pred_stage_correct)
            f_prec, f_rec, f_f1 = calc_metrics(y_true, y_pred_stage_filter)

            filtered_regions.to_csv(os.path.join(args.data_dir, f"{held_out}_t{thresh}_final_prediction.bed"), sep="\t", header=False, index=False)

            metrics_log.append({
                "sample_id": held_out,
                "threshold": thresh,
                "bin_pr_auc": pr_auc,
                "bin_roc_auc": roc_auc,
                "raw_precision": r_prec,    "raw_recall": r_rec,    "raw_f1": r_f1,
                "merge_precision": m_prec,  "merge_recall": m_rec,  "merge_f1": m_f1,
                "correct_precision": c_prec, "correct_recall": c_rec, "correct_f1": c_f1,
                "filter_precision": f_prec,  "filter_recall": f_rec,  "filter_f1": f_f1
            })

            print(f"Raw      : F1: {r_f1:.3f}  Prec: {r_prec:.3f}  Rec: {r_rec:.3f}")
            print(f"Merged    : F1: {m_f1:.3f}  Prec: {m_prec:.3f}  Rec: {m_rec:.3f}")
            print(f"Corrected : F1: {c_f1:.3f}  Prec: {c_prec:.3f}  Rec: {c_rec:.3f}")
            print(f"Filtered  : F1: {f_f1:.3f}  Prec: {f_prec:.3f}  Rec: {f_rec:.3f}")

        del model, test_df, called_bins, raw_regions, length_filtered_regions, merged_regions, corrected_regions, filtered_regions; gc.collect()

    print("\n Training final model")
    all_final_dfs = [train_data[s] for s in samples]
    final_model = train_universal_model(all_final_dfs, sig_feats, args)

    if final_model:
        final_model_path = os.path.join(args.data_dir, "termination_region_prediction.json")
        final_model.save_model(final_model_path)
        feature_list_path = os.path.join(args.data_dir, "termination_region_prediction_feature_list.json")
        with open(feature_list_path, "w") as f:
            json.dump(sig_feats, f)

    for sample_id, path in parquet_paths.items():
        if os.path.exists(path):
            os.remove(path)

    pd.DataFrame(metrics_log).to_csv(os.path.join(args.data_dir, "LOCO_metrics_summary.csv"), index=False)

if __name__ == "__main__":
    main()
