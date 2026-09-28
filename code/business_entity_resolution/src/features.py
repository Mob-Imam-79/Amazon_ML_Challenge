"""
features.py — Pairwise feature engineering for entity matching.
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from rapidfuzz.distance import JaroWinkler
import jellyfish
import time


def _safe_str(val):
    """Convert to string, handle NaN/None."""
    if pd.isna(val) or val is None:
        return ''
    return str(val)


def _jaccard_ngrams(s1: str, s2: str, n: int = 3) -> float:
    """Jaccard similarity on character n-grams."""
    if not s1 or not s2:
        return 0.0
    set1 = set(s1[i:i+n] for i in range(len(s1)-n+1))
    set2 = set(s2[i:i+n] for i in range(len(s2)-n+1))
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def _token_overlap(s1: str, s2: str) -> float:
    """Fraction of tokens shared between two strings."""
    if not s1 or not s2:
        return 0.0
    t1 = set(s1.split())
    t2 = set(s2.split())
    if not t1 or not t2:
        return 0.0
    intersection = len(t1 & t2)
    union = len(t1 | t2)
    return intersection / union if union > 0 else 0.0


def _common_prefix_len(s1: str, s2: str) -> int:
    """Length of the common prefix."""
    if not s1 or not s2:
        return 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return i
    return min(len(s1), len(s2))


def _number_match(nums1: str, nums2: str) -> float:
    """Fraction of numeric tokens that match."""
    if not nums1 or not nums2:
        return 0.0
    set1 = set(nums1.split())
    set2 = set(nums2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / max(len(set1), len(set2))


def _len_ratio(s1: str, s2: str) -> float:
    """Ratio of lengths (shorter/longer)."""
    l1, l2 = len(s1), len(s2)
    if l1 == 0 and l2 == 0:
        return 1.0
    if l1 == 0 or l2 == 0:
        return 0.0
    return min(l1, l2) / max(l1, l2)

def compute_pair_features(row) -> dict:
    """
    Compute all pairwise features for a candidate pair.

    Expects row with: name_norm_s1, name_norm_cand, addr_norm_s1, addr_norm_cand,
                      addr_numbers_s1, addr_numbers_cand, tfidf_score
    """
    name1 = _safe_str(row.get('name_norm_s1', ''))
    name2 = _safe_str(row.get('name_norm_cand', ''))
    addr1 = _safe_str(row.get('addr_norm_s1', ''))
    addr2 = _safe_str(row.get('addr_norm_cand', ''))
    nums1 = _safe_str(row.get('addr_numbers_s1', ''))
    nums2 = _safe_str(row.get('addr_numbers_cand', ''))

    features = {}

    # ── Name features ─────────────────────────────────────────────────────
    features['name_jaro_winkler'] = JaroWinkler.similarity(name1, name2) if name1 and name2 else 0.0
    features['name_levenshtein'] = fuzz.ratio(name1, name2) / 100.0
    features['name_token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100.0
    features['name_token_set'] = fuzz.token_set_ratio(name1, name2) / 100.0
    features['name_partial'] = fuzz.partial_ratio(name1, name2) / 100.0
    features['name_jaccard_3gram'] = _jaccard_ngrams(name1, name2, 3)
    features['name_jaccard_4gram'] = _jaccard_ngrams(name1, name2, 4)
    features['name_token_overlap'] = _token_overlap(name1, name2)
    features['name_prefix_len'] = _common_prefix_len(name1, name2)
    features['name_len_ratio'] = _len_ratio(name1, name2)
    features['name_len_diff'] = abs(len(name1) - len(name2))

    # ── Address features ──────────────────────────────────────────────────
    features['addr_levenshtein'] = fuzz.ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    features['addr_token_sort'] = fuzz.token_sort_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    features['addr_token_set'] = fuzz.token_set_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    features['addr_token_overlap'] = _token_overlap(addr1, addr2)
    features['addr_number_match'] = _number_match(nums1, nums2)
    features['addr_len_ratio'] = _len_ratio(addr1, addr2)
    features['addr_missing'] = int(not addr1 or not addr2)

    # ── Combined features ─────────────────────────────────────────────────
    features['tfidf_score'] = row.get('tfidf_score', 0.0)

    # Avg and max of name + addr
    name_scores = [features['name_jaro_winkler'], features['name_token_sort'],
                   features['name_token_set']]
    addr_scores = [features['addr_levenshtein'], features['addr_token_sort'],
                   features['addr_token_set']]

    features['name_avg_sim'] = np.mean(name_scores)
    features['name_max_sim'] = np.max(name_scores)
    features['addr_avg_sim'] = np.mean(addr_scores) if not features['addr_missing'] else 0.0
    features['combined_score'] = 0.6 * features['name_avg_sim'] + 0.4 * features['addr_avg_sim']

    return features


FEATURE_COLUMNS = [
    'name_jaro_winkler', 'name_levenshtein', 'name_token_sort', 'name_token_set',
    'name_partial', 'name_jaccard_3gram', 'name_jaccard_4gram', 'name_token_overlap',
    'name_prefix_len', 'name_len_ratio', 'name_len_diff',
    'addr_levenshtein', 'addr_token_sort', 'addr_token_set', 'addr_token_overlap',
    'addr_number_match', 'addr_len_ratio', 'addr_missing',
    'tfidf_score',
    'name_avg_sim', 'name_max_sim', 'addr_avg_sim', 'combined_score',
]


def compute_features_batch(candidates_df, s1_df, s2s3_df, batch_size=100_000):
    """
    Compute features for all candidate pairs in batches.

    Args:
        candidates_df: DataFrame with [s1_entity_id, candidate_entity_id, tfidf_score]
        s1_df:         S1 DataFrame (preprocessed)
        s2s3_df:       S2+S3 DataFrame (preprocessed)
        batch_size:    Process this many pairs at a time

    Returns:
        DataFrame with all features + entity IDs
    """
    t0 = time.time()
    n_total = len(candidates_df)
    print(f"\n  ── Computing features for {n_total:,} candidate pairs ──")

    # Build lookup dicts for fast access
    print("    Building lookup indices...")
    s1_lookup = s1_df.set_index('entity_id')[['name_norm', 'addr_norm', 'addr_numbers']].to_dict('index')
    s2s3_lookup = s2s3_df.set_index('entity_id')[['name_norm', 'addr_norm', 'addr_numbers']].to_dict('index')

    all_features = []
    n_batches = (n_total + batch_size - 1) // batch_size

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_total)
        batch = candidates_df.iloc[start:end]

        batch_s1_ids = batch['s1_entity_id'].values
        batch_cand_ids = batch['candidate_entity_id'].values
        batch_tfidf = batch['tfidf_score'].values

        batch_features = []
        for s1_id, cand_id, tfidf in zip(batch_s1_ids, batch_cand_ids, batch_tfidf):
            s1_data = s1_lookup.get(s1_id, {})
            cand_data = s2s3_lookup.get(cand_id, {})

            feat_row = {
                'name_norm_s1': s1_data.get('name_norm', ''),
                'name_norm_cand': cand_data.get('name_norm', ''),
                'addr_norm_s1': s1_data.get('addr_norm', ''),
                'addr_norm_cand': cand_data.get('addr_norm', ''),
                'addr_numbers_s1': s1_data.get('addr_numbers', ''),
                'addr_numbers_cand': cand_data.get('addr_numbers', ''),
                'tfidf_score': tfidf,
            }

            features = compute_pair_features(feat_row)
            features['s1_entity_id'] = s1_id
            features['candidate_entity_id'] = cand_id
            batch_features.append(features)

        all_features.extend(batch_features)

        if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
            elapsed = time.time() - t0
            pct = 100 * (end) / n_total
            rate = end / elapsed if elapsed > 0 else 0
            print(f"    Batch {batch_idx+1}/{n_batches} ({pct:.1f}%) — "
                  f"{rate:.0f} pairs/sec — {elapsed:.1f}s elapsed")

    result = pd.DataFrame(all_features)
    elapsed = time.time() - t0
    print(f"  ── Features complete: {len(result):,} rows × {len(FEATURE_COLUMNS)} features in {elapsed:.1f}s ──")

    return result
