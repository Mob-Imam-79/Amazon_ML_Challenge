"""
model.py — LightGBM training, threshold tuning, and prediction.

Key design: Train/val split is at the S1 ENTITY level (not pair level)
to avoid data leakage and get accurate F_0.5 estimates.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import time
import os
import json

from src.features import FEATURE_COLUMNS


def create_labels(features_df, ground_truth_df):
    """
    Create binary labels for candidate pairs based on ground truth.

    Args:
        features_df:      DataFrame with [s1_entity_id, candidate_entity_id, ...features...]
        ground_truth_df:  DataFrame with [source1_entity_id, matched_entity_ids]

    Returns:
        numpy array of labels (1=match, 0=no match)
    """
    print("  Creating labels from ground truth...")

    # Build a set of (s1_id, matched_id) pairs for fast lookup
    gt_pairs = set()
    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ''
        if matched.strip():
            for mid in matched.split(','):
                mid = mid.strip()
                if mid:
                    gt_pairs.add((s1_id, mid))

    labels = np.zeros(len(features_df), dtype=np.int32)
    for i in range(len(features_df)):
        pair = (features_df.iloc[i]['s1_entity_id'], features_df.iloc[i]['candidate_entity_id'])
        if pair in gt_pairs:
            labels[i] = 1

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    print(f"    Positive pairs: {n_pos:,} ({100*n_pos/len(labels):.2f}%)")
    print(f"    Negative pairs: {n_neg:,} ({100*n_neg/len(labels):.2f}%)")

    return labels


def entity_level_split(features_df, labels, ground_truth_df, val_size=0.2, random_state=42):
    """
    Split data at the S1 ENTITY level to avoid data leakage.
    All candidate pairs for a given S1 entity go to either train OR val, never both.

    Returns:
        (X_train, y_train, X_val, y_val, val_entity_ids, val_features_df)
    """
    print("\n  Performing entity-level train/val split...")

    # Get unique S1 entities with their country for stratification
    s1_entities = features_df[['s1_entity_id']].drop_duplicates()

    # Merge country from ground truth (if available) for stratification
    train_ids, val_ids = train_test_split(
        s1_entities['s1_entity_id'].values,
        test_size=val_size,
        random_state=random_state,
    )

    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)

    train_mask = features_df['s1_entity_id'].isin(train_ids_set).values
    val_mask = features_df['s1_entity_id'].isin(val_ids_set).values

    X_train = features_df.loc[train_mask, FEATURE_COLUMNS].values
    y_train = labels[train_mask]
    X_val = features_df.loc[val_mask, FEATURE_COLUMNS].values
    y_val = labels[val_mask]

    val_features_df = features_df.loc[val_mask].reset_index(drop=True)

    print(f"    Train entities: {len(train_ids):,} → {len(X_train):,} pairs ({y_train.sum():,} pos)")
    print(f"    Val entities:   {len(val_ids):,} → {len(X_val):,} pairs ({y_val.sum():,} pos)")

    return X_train, y_train, X_val, y_val, val_ids_set, val_features_df


def train_model(features_df, labels, ground_truth_df, val_size=0.2, random_state=42):
    """
    Train a LightGBM classifier with entity-level split.

    Returns:
        (model, X_val, y_val, val_entity_ids, val_features_df)
    """
    t0 = time.time()
    print("\n  ── Training LightGBM model ──")

    # Entity-level split
    X_train, y_train, X_val, y_val, val_entity_ids, val_features_df = \
        entity_level_split(features_df, labels, ground_truth_df, val_size, random_state)

    # Class weight
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"    Scale pos weight: {pos_weight:.2f}")

    # Train
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        max_depth=8,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,
        scale_pos_weight=pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_X=X_val, eval_y=y_val,
        eval_metric='binary_logloss',
        callbacks=[
            lgb.early_stopping(50, verbose=True),
            lgb.log_evaluation(100),
        ],
    )

    # Feature importance
    importance = pd.DataFrame({
        'feature': FEATURE_COLUMNS,
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False)

    print("\n    Top-10 features:")
    for _, r in importance.head(10).iterrows():
        print(f"      {r['feature']:25s}  {r['importance']:>6.0f}")

    elapsed = time.time() - t0
    print(f"\n  ── Training complete in {elapsed:.1f}s ──")

    return model, X_val, y_val, val_entity_ids, val_features_df


def compute_f05_macro(gt_dict, predictions_dict, all_s1_ids):
    """
    Compute macro-averaged F_0.5 across ALL S1 entities (including singletons).

    Args:
        gt_dict:          {s1_id: set(matched_ids)} from ground truth
        predictions_dict: {s1_id: set(predicted_ids)} from model
        all_s1_ids:       set of all S1 entity IDs to evaluate

    Returns:
        macro_f05 score
    """
    f05_scores = []

    for s1_id in all_s1_ids:
        true_set = gt_dict.get(s1_id, set())
        pred_set = predictions_dict.get(s1_id, set())

        if len(pred_set) == 0 and len(true_set) == 0:
            f05_scores.append(1.0)  # Correct singleton
        elif len(pred_set) == 0 and len(true_set) > 0:
            f05_scores.append(0.0)  # Missed all matches
        elif len(pred_set) > 0 and len(true_set) == 0:
            f05_scores.append(0.0)  # False positive on singleton
        else:
            precision = len(true_set & pred_set) / len(pred_set)
            recall = len(true_set & pred_set) / len(true_set)
            if precision + recall == 0:
                f05_scores.append(0.0)
            else:
                f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
                f05_scores.append(f05)

    return np.mean(f05_scores)


def find_optimal_threshold(model, X_val, y_val, val_features_df,
                           ground_truth_df, val_entity_ids):
    """
    Find the optimal prediction threshold to maximize F_0.5 (macro-averaged).
    Evaluates on ALL validation entities, including singletons with no candidates.

    Returns:
        (best_threshold, best_f05)
    """
    print("\n  ── Finding optimal threshold ──")

    probs = model.predict_proba(X_val)[:, 1]

    val_df = val_features_df.copy()
    val_df['prob'] = probs

    # Build ground truth lookup
    gt_dict = {}
    for _, row in ground_truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ''
        if matched.strip():
            gt_dict[s1_id] = set(m.strip() for m in matched.split(',') if m.strip())
        else:
            gt_dict[s1_id] = set()

    best_threshold = 0.5
    best_f05 = 0.0
    results = []

    for threshold in np.arange(0.10, 0.96, 0.02):
        # Build predictions for each val entity
        predictions_dict = {}
        for s1_id, group in val_df.groupby('s1_entity_id'):
            matched = set(group[group['prob'] >= threshold]['candidate_entity_id'].values)
            predictions_dict[s1_id] = matched

        # Also include val entities with NO candidates (singletons)
        for s1_id in val_entity_ids:
            if s1_id not in predictions_dict:
                predictions_dict[s1_id] = set()

        macro_f05 = compute_f05_macro(gt_dict, predictions_dict, val_entity_ids)
        results.append({'threshold': threshold, 'f05': macro_f05})

        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_threshold = threshold

    print(f"    Best threshold: {best_threshold:.2f}")
    print(f"    Best F_0.5:     {best_f05:.4f}")

    # Print surrounding thresholds
    results_df = pd.DataFrame(results)
    near_best = results_df[abs(results_df['threshold'] - best_threshold) <= 0.12]
    print("\n    Threshold sweep around optimal:")
    for _, r in near_best.iterrows():
        marker = " ◀" if abs(r['threshold'] - best_threshold) < 0.01 else ""
        print(f"      t={r['threshold']:.2f}  F₀.₅={r['f05']:.4f}{marker}")

    return best_threshold, best_f05


def predict_matches(model, features_df, threshold):
    """
    Generate final match predictions.

    Returns:
        DataFrame with [source1_entity_id, matched_entity_ids]
    """
    print(f"\n  ── Predicting matches (threshold={threshold:.2f}) ──")

    X = features_df[FEATURE_COLUMNS].values
    probs = model.predict_proba(X)[:, 1]

    features_df = features_df.copy()
    features_df['prob'] = probs

    # Group by S1 entity and collect matches above threshold
    matches = {}
    for s1_id, group in features_df.groupby('s1_entity_id'):
        matched = group[group['prob'] >= threshold]['candidate_entity_id'].values
        matches[s1_id] = ','.join(matched) if len(matched) > 0 else ''

    result = pd.DataFrame([
        {'source1_entity_id': k, 'matched_entity_ids': v}
        for k, v in matches.items()
    ])

    n_matched = (result['matched_entity_ids'] != '').sum()
    n_singleton = (result['matched_entity_ids'] == '').sum()
    print(f"    Matched entities:   {n_matched:,}")
    print(f"    Singleton entities: {n_singleton:,}")

    return result


def save_model(model, threshold, output_dir):
    """Save model and metadata."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.txt')
    model.booster_.save_model(model_path)

    meta = {'threshold': threshold, 'features': FEATURE_COLUMNS}
    with open(os.path.join(output_dir, 'model_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Model saved to {output_dir}")
