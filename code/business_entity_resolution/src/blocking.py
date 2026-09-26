"""
blocking.py — Candidate generation using TF-IDF character n-grams + sparse cosine similarity.

Memory-optimized for M2 Mac 16GB: chunks large S2/S3 sets to avoid OOM.
"""

import gc
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import awesome_cossim_topn
import time


def tfidf_blocking(df_s1, df_s2s3, country,
                   ngram_range=(3, 5), top_n=15, threshold=0.30,
                   max_features=100_000, chunk_size=1_000_000):
    """
    Perform TF-IDF character n-gram blocking for a specific country.
    Chunks S2/S3 into batches of `chunk_size` to control memory.

    Args:
        df_s1:        S1 DataFrame (filtered to one country)
        df_s2s3:      S2+S3 DataFrame (filtered to same country)
        country:      Country string (for logging)
        ngram_range:  Character n-gram range for TF-IDF
        top_n:        Max candidates per S1 entity
        threshold:    Minimum cosine similarity to keep
        max_features: Max TF-IDF vocabulary size (controls memory)
        chunk_size:   Max S2/S3 records per chunk

    Returns:
        List of (s1_idx, s2s3_idx, cosine_score) tuples
    """
    t0 = time.time()
    n_s1 = len(df_s1)
    n_s2s3 = len(df_s2s3)
    print(f"    TF-IDF blocking [{country}]: {n_s1:,} S1 × {n_s2s3:,} S2/S3...")

    if n_s1 == 0 or n_s2s3 == 0:
        print(f"      ⚠ Skipping (empty set)")
        return []

    # Determine if chunking is needed
    n_chunks = max(1, (n_s2s3 + chunk_size - 1) // chunk_size)

    if n_chunks > 1:
        print(f"      Splitting S2/S3 into {n_chunks} chunks of ~{chunk_size:,} for memory safety")

    # Fit vectorizer on S1 only (consistent vocabulary across chunks)
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=ngram_range,
        max_features=max_features,
        dtype=np.float32,
        sublinear_tf=True,
    )

    tfidf_s1 = vectorizer.fit_transform(df_s1['combined_text'])
    print(f"      TF-IDF S1 matrix: {tfidf_s1.shape}, vocab: {len(vectorizer.vocabulary_):,}")

    all_candidates = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n_s2s3)
        chunk_s2s3 = df_s2s3.iloc[start:end]

        tfidf_chunk = vectorizer.transform(chunk_s2s3['combined_text'])

        # Sparse top-N cosine similarity
        cosine_matrix = awesome_cossim_topn(
            tfidf_s1, tfidf_chunk.T,
            ntop=top_n,
            lower_bound=threshold,
            use_threads=True,
            n_jobs=4,
        )

        # Extract results — offset j by chunk start index
        coo = cosine_matrix.tocoo()
        for i, j, v in zip(coo.row, coo.col, coo.data):
            all_candidates.append((i, start + j, float(v)))

        if n_chunks > 1:
            print(f"        Chunk {chunk_idx+1}/{n_chunks}: {coo.nnz:,} pairs")

        # Free chunk memory
        del tfidf_chunk, cosine_matrix, coo
        gc.collect()

    # Deduplicate: keep top_n per S1 entity across all chunks
    if n_chunks > 1 and len(all_candidates) > 0:
        from collections import defaultdict
        per_s1 = defaultdict(list)
        for i, j, v in all_candidates:
            per_s1[i].append((j, v))

        deduped = []
        for i, pairs in per_s1.items():
            pairs.sort(key=lambda x: -x[1])
            for j, v in pairs[:top_n]:
                deduped.append((i, j, v))
        all_candidates = deduped

    elapsed = time.time() - t0
    print(f"      ✓ {len(all_candidates):,} candidate pairs in {elapsed:.1f}s "
          f"(avg {len(all_candidates)/max(n_s1,1):.1f} per S1 entity)")

    # Free S1 TF-IDF
    del tfidf_s1
    gc.collect()

    return all_candidates


def run_blocking(s1_df, s2s3_df, top_n=15, threshold=0.30,
                 max_features=100_000, chunk_size=1_000_000):
    """
    Run TF-IDF blocking across all countries.

    Args:
        s1_df:         Full S1 DataFrame (preprocessed)
        s2s3_df:       Full S2+S3 DataFrame (preprocessed)
        top_n:         Max candidates per S1 entity
        threshold:     Minimum cosine similarity
        max_features:  TF-IDF vocabulary size cap
        chunk_size:    Max S2/S3 records per blocking chunk

    Returns:
        DataFrame with columns: [s1_entity_id, candidate_entity_id, tfidf_score]
    """
    t0 = time.time()
    print("\n  ── Running blocking ──")

    all_countries = sorted(set(s1_df['country'].unique()) | set(s2s3_df['country'].unique()))
    print(f"  Countries: {all_countries}")

    all_candidates = []

    for country in all_countries:
        s1_country = s1_df[s1_df['country'] == country].reset_index(drop=True)
        s2s3_country = s2s3_df[s2s3_df['country'] == country].reset_index(drop=True)

        if len(s1_country) == 0:
            print(f"    ⚠ No S1 entities for {country}, skipping")
            continue

        cands = tfidf_blocking(s1_country, s2s3_country, country,
                               top_n=top_n, threshold=threshold,
                               max_features=max_features, chunk_size=chunk_size)

        # Map back to entity_ids
        s1_ids = s1_country['entity_id'].values
        s2s3_ids = s2s3_country['entity_id'].values

        for s1_idx, s2s3_idx, score in cands:
            all_candidates.append({
                's1_entity_id': s1_ids[s1_idx],
                'candidate_entity_id': s2s3_ids[s2s3_idx],
                'tfidf_score': score,
            })

        # Free country slices
        del s1_country, s2s3_country, cands
        gc.collect()

    result = pd.DataFrame(all_candidates)
    elapsed = time.time() - t0
    print(f"\n  ── Blocking complete: {len(result):,} total candidates in {elapsed:.1f}s ──")

    if len(result) > 0:
        n_s1_with_cands = result['s1_entity_id'].nunique()
        avg_cands = len(result) / n_s1_with_cands
        print(f"  S1 entities with candidates: {n_s1_with_cands:,} / {len(s1_df):,}")
        print(f"  Average candidates per S1: {avg_cands:.1f}")

    return result
