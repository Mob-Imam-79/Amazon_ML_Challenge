#!/usr/bin/env python3
"""
pipeline.py — End-to-end Entity Resolution Pipeline
=====================================================
Usage:
    # Quick dev run on 10K sample:
    python pipeline.py --mode train --sample 10000

    # Full training run:
    python pipeline.py --mode train

    # Generate test predictions (after training):
    python pipeline.py --mode predict

    # Full end-to-end (train + predict):
    python pipeline.py --mode full --sample 50000
"""

import os
import sys
import time
import argparse
import pickle

import numpy as np
import pandas as pd

# Add src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)  # business_entity_resolution/
sys.path.insert(0, PACKAGE_DIR)

from src.preprocess import preprocess_dataframe
from src.blocking import run_blocking
from src.features import compute_features_batch, FEATURE_COLUMNS
from src.model import (
    create_labels, train_model, find_optimal_threshold,
    predict_matches, save_model,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(PACKAGE_DIR))  # student_resource/
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'dataset', 'test')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
MODEL_DIR  = os.path.join(SCRIPT_DIR, 'model_artifacts')


def hr(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def load_data(data_dir, prefix='train'):
    """Load source TSV files."""
    s1 = pd.read_csv(os.path.join(data_dir, f'{prefix}_source1.tsv'), sep='\t')
    s2 = pd.read_csv(os.path.join(data_dir, f'{prefix}_source2.tsv'), sep='\t')
    s3 = pd.read_csv(os.path.join(data_dir, f'{prefix}_source3.tsv'), sep='\t')
    return s1, s2, s3


def run_train(sample_size=None, blocking_topn=20, blocking_threshold=0.25):
    """
    Full training pipeline:
      1. Load & preprocess training data
      2. Run TF-IDF blocking to generate candidates
      3. Compute pairwise features
      4. Train LightGBM model
      5. Tune threshold for F_0.5
      6. Save model
    """
    total_t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────
    hr("1 · Loading training data")
    s1, s2, s3 = load_data(TRAIN_DIR, 'train')
    gt = pd.read_csv(os.path.join(TRAIN_DIR, 'train_ground_truth.tsv'), sep='\t')
    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}  GT: {len(gt):,}")

    # ── Sample if requested ──────────────────────────────────────────────
    if sample_size and sample_size < len(s1):
        hr(f"  Sampling {sample_size:,} S1 entities for dev run")
        s1_sample = s1.sample(n=sample_size, random_state=42)
        gt = gt[gt['source1_entity_id'].isin(s1_sample['entity_id'])]
        s1 = s1_sample.reset_index(drop=True)

        # Also filter S2/S3 to only keep entities that appear in GT matches
        matched_ids = set()
        for _, row in gt.iterrows():
            if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip():
                for mid in str(row['matched_entity_ids']).split(','):
                    matched_ids.add(mid.strip())

        # Keep all S2/S3 from same countries (needed for blocking)
        countries = s1['country'].unique()
        s2 = s2[s2['country'].isin(countries)].reset_index(drop=True)
        s3 = s3[s3['country'].isin(countries)].reset_index(drop=True)

        # Sub-sample S2/S3 to keep it manageable: keep matched + random sample
        s2_matched = s2[s2['entity_id'].isin(matched_ids)]
        s3_matched = s3[s3['entity_id'].isin(matched_ids)]
        s2_ratio = min(1.0, 3 * sample_size / len(s2))
        s3_ratio = min(1.0, 3 * sample_size / len(s3))
        s2_random = s2[~s2['entity_id'].isin(matched_ids)].sample(
            frac=s2_ratio, random_state=42)
        s3_random = s3[~s3['entity_id'].isin(matched_ids)].sample(
            frac=s3_ratio, random_state=42)
        s2 = pd.concat([s2_matched, s2_random]).drop_duplicates('entity_id').reset_index(drop=True)
        s3 = pd.concat([s3_matched, s3_random]).drop_duplicates('entity_id').reset_index(drop=True)

        print(f"  After sampling: S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  GT={len(gt):,}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    hr("2 · Preprocessing")
    s1 = preprocess_dataframe(s1, 'Train S1')
    s2 = preprocess_dataframe(s2, 'Train S2')
    s3 = preprocess_dataframe(s3, 'Train S3')

    # Combine S2 + S3
    s2s3 = pd.concat([s2, s3], ignore_index=True)
    print(f"  Combined S2+S3: {len(s2s3):,} records")

    # ── 3. Blocking ───────────────────────────────────────────────────────
    hr("3 · Blocking (TF-IDF)")
    candidates = run_blocking(s1, s2s3, top_n=blocking_topn, threshold=blocking_threshold)

    if len(candidates) == 0:
        print("  ❌ No candidates generated! Adjust blocking parameters.")
        return

    # ── Measure blocking recall ───────────────────────────────────────────
    gt_pairs = set()
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        if pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip():
            for mid in str(row['matched_entity_ids']).split(','):
                gt_pairs.add((s1_id, mid.strip()))

    cand_pairs = set(zip(candidates['s1_entity_id'], candidates['candidate_entity_id']))
    found = len(gt_pairs & cand_pairs)
    total_gt = len(gt_pairs)
    blocking_recall = found / total_gt if total_gt > 0 else 0
    print(f"\n  ⚡ Blocking recall: {found:,}/{total_gt:,} = {blocking_recall:.4f}")
    print(f"     (This is the recall ceiling for the pipeline)")

    # ── 4. Feature engineering ────────────────────────────────────────────
    hr("4 · Feature engineering")
    features_df = compute_features_batch(candidates, s1, s2s3)

    # ── 5. Train model ────────────────────────────────────────────────────
    hr("5 · Training LightGBM")
    labels = create_labels(features_df, gt)
    model, X_val, y_val, val_entity_ids, val_features_df = train_model(
        features_df, labels, gt)

    # ── 6. Threshold tuning ───────────────────────────────────────────────
    hr("6 · Threshold tuning")
    best_threshold, best_f05 = find_optimal_threshold(
        model, X_val, y_val, val_features_df, gt, val_entity_ids)

    # ── 7. Save model ─────────────────────────────────────────────────────
    hr("7 · Saving model")
    save_model(model, best_threshold, MODEL_DIR)

    # Also save as pickle for easy loading
    with open(os.path.join(MODEL_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'threshold': best_threshold}, f)

    total_elapsed = time.time() - total_t0
    hr("✅ Training complete")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"  Blocking recall: {blocking_recall:.4f}")
    print(f"  Best F₀.₅ (val): {best_f05:.4f}")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"  Model saved to: {MODEL_DIR}")

    return model, best_threshold


def run_predict(model=None, threshold=None):
    """
    Generate predictions on test data:
      1. Load & preprocess test data
      2. Run blocking
      3. Compute features
      4. Predict matches using trained model
      5. Write output files
    """
    total_t0 = time.time()

    # ── Load model if not provided ────────────────────────────────────────
    if model is None or threshold is None:
        hr("Loading saved model")
        model_path = os.path.join(MODEL_DIR, 'model.pkl')
        if not os.path.exists(model_path):
            print(f"  ❌ No model found at {model_path}. Run training first.")
            return
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
        model = saved['model']
        threshold = saved['threshold']
        print(f"  Loaded model with threshold={threshold:.2f}")

    # ── 1. Load test data ─────────────────────────────────────────────────
    hr("1 · Loading test data")
    s1, s2, s3 = load_data(TEST_DIR, 'test')
    print(f"  S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}")

    # ── 2. Preprocess ─────────────────────────────────────────────────────
    hr("2 · Preprocessing")
    s1 = preprocess_dataframe(s1, 'Test S1')
    s2 = preprocess_dataframe(s2, 'Test S2')
    s3 = preprocess_dataframe(s3, 'Test S3')

    s2s3 = pd.concat([s2, s3], ignore_index=True)
    print(f"  Combined S2+S3: {len(s2s3):,} records")

    # ── 3. Blocking ───────────────────────────────────────────────────────
    hr("3 · Blocking (TF-IDF)")
    candidates = run_blocking(s1, s2s3, top_n=20, threshold=0.25)

    # ── 4. Features ───────────────────────────────────────────────────────
    hr("4 · Feature engineering")
    features_df = compute_features_batch(candidates, s1, s2s3)

    # ── 5. Predict ────────────────────────────────────────────────────────
    hr("5 · Predicting matches")
    results = predict_matches(model, features_df, threshold)

    # ── 6. Ensure all S1 entities are present ─────────────────────────────
    all_s1_ids = set(s1['entity_id'].values)
    present_ids = set(results['source1_entity_id'].values)
    missing_ids = all_s1_ids - present_ids

    if missing_ids:
        print(f"  Adding {len(missing_ids):,} S1 entities with no candidates (singletons)")
        missing_rows = pd.DataFrame({
            'source1_entity_id': list(missing_ids),
            'matched_entity_ids': '',
        })
        results = pd.concat([results, missing_rows], ignore_index=True)

    results = results.sort_values('source1_entity_id').reset_index(drop=True)

    # ── 7. Write outputs ──────────────────────────────────────────────────
    hr("6 · Writing output files")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # matching_results.tsv
    matching_path = os.path.join(OUTPUT_DIR, 'matching_results.tsv')
    results.to_csv(matching_path, sep='\t', index=False)
    print(f"  ✅ {matching_path}")

    # candidate_pairs.tsv
    cand_grouped = candidates.groupby('s1_entity_id')['candidate_entity_id'].apply(
        lambda x: ','.join(x.values)
    ).reset_index()
    cand_grouped.columns = ['source1_entity_id', 'candidate_entity_ids']

    # Add singletons
    cand_present = set(cand_grouped['source1_entity_id'].values)
    cand_missing = all_s1_ids - cand_present
    if cand_missing:
        cand_missing_rows = pd.DataFrame({
            'source1_entity_id': list(cand_missing),
            'candidate_entity_ids': '',
        })
        cand_grouped = pd.concat([cand_grouped, cand_missing_rows], ignore_index=True)

    cand_grouped = cand_grouped.sort_values('source1_entity_id').reset_index(drop=True)
    cand_path = os.path.join(OUTPUT_DIR, 'candidate_pairs.tsv')
    cand_grouped.to_csv(cand_path, sep='\t', index=False)
    print(f"  ✅ {cand_path}")

    total_elapsed = time.time() - total_t0
    hr("✅ Prediction complete")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print(f"  Total S1 entities: {len(results):,}")
    print(f"  Matched: {(results['matched_entity_ids'] != '').sum():,}")
    print(f"  Singletons: {(results['matched_entity_ids'] == '').sum():,}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Entity Resolution Pipeline')
    parser.add_argument('--mode', choices=['train', 'predict', 'full'],
                        default='full', help='Pipeline mode')
    parser.add_argument('--sample', type=int, default=None,
                        help='Sample N S1 entities for dev run (e.g., 10000)')
    parser.add_argument('--blocking-topn', type=int, default=15,
                        help='Max candidates per S1 entity from blocking')
    parser.add_argument('--blocking-threshold', type=float, default=0.30,
                        help='Min TF-IDF cosine similarity for blocking')
    args = parser.parse_args()

    print(f"\n  Pipeline mode: {args.mode}")
    if args.sample:
        print(f"  Sample size: {args.sample:,}")
    print()

    model, threshold = None, None

    if args.mode in ('train', 'full'):
        model, threshold = run_train(
            sample_size=args.sample,
            blocking_topn=args.blocking_topn,
            blocking_threshold=args.blocking_threshold,
        )

    if args.mode in ('predict', 'full'):
        run_predict(model=model, threshold=threshold)


if __name__ == '__main__':
    main()
