# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team EntityResolvers (Mobashir & Team)  
**Team Members:** Md Mobashir Imam  
**Submission Date:** September 2026  

---

## 1. Executive Summary

We developed an end-to-end, high-performance Business Entity Resolution pipeline engineered specifically for the precision-weighted macro $F_{0.5}$ metric on multi-source noisy commercial datasets. Our solution employs country-partitioned TF-IDF character n-gram blocking with sublinear term frequency and chunked top-$N$ sparse cosine similarity, capturing a 95.58% recall ceiling while drastically pruning comparisons. Candidate pairs are classified by a LightGBM gradient boosted tree model trained on 23 pairwise name, address, and composite string similarity features using a strictly disjoint entity-level validation split to prevent data leakage. By tuning the decision threshold directly against the macro $F_{0.5}$ objective across both matched entities and singletons, our model achieves a validation macro $F_{0.5}$ score of **0.9474** at an optimal threshold of **0.88**.

---

## 2. Methodology

### 2.1 Problem Analysis
Through rigorous Exploratory Data Analysis (EDA) across 2.2M training records and 1.73M test records across Sources 1, 2, and 3, we uncovered key characteristics that directly influenced our pipeline design:
1. **Severe Text & Format Heterogeneity**: Business names exhibit extensive abbreviation differences (e.g., *Pvt Ltd* vs. *Private Limited*, *Corp* vs. *Corporation*, *Inc* vs. *Incorporated*), DBA ("doing business as") prefixes, URL/domain artifacts, and punctuation shifts (& vs. "and").
2. **Address Variations & Landmark References**: Addresses frequently omit state codes or postal PIN codes, invert municipal numbering vs. street names, or rely on colloquial landmarks (e.g., "Near SBI ATM", "Opposite City Hospital"). Furthermore, non-ASCII Unicode characters and regional transliterations are prevalent.
3. **Open-World Country Set**: While training data covers `US` and `India`, test data introduces `France` (over 259,000 Source 1 test records). Models and tokenizers must operate purely on character/token patterns without hard-coding country taxonomies.
4. **Prevalence of Singletons**: In the ground truth, approximately **4.1%** of Source 1 entities have zero matching records in Sources 2 and 3. Under the competition macro $F_{0.5}$ scoring, correctly predicting an empty match list yields a perfect 1.0 score, whereas any false positive merge on a singleton drops the score to 0.0. Conservative, high-precision matching is therefore paramount.
5. **Multi-Source Matches**: For entities with matches, 22.8% match records in both Source 2 and Source 3, with an average of 1.96 matches per matched entity.

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Sublinear TF-IDF Blocking + 23-Feature Engineering + Leak-Free LightGBM Classifier with Metric-Aligned Threshold Optimization  
**Core Innovation:** 
- *Zero-Leakage Entity Split*: Candidate pairs are partitioned strictly by `source1_entity_id`, ensuring all candidates of an entity remain in either train or val, providing true out-of-sample generalization.
- *Metric-Aligned Objective*: Calibration of the classification threshold directly against the macro-averaged $F_{0.5}$ metric (which weights precision 2× over recall), preventing catastrophic false merges on both matched entities and singletons.
- *Memory-Bounded Chunked Sparse Cosine Top-N*: Sublinear character 3–5 n-gram blocking with batched dot-products that scales smoothly on commodity hardware without out-of-memory (OOM) failures.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73\text{M} \times 9.97\text{M} \approx 1.72 \times 10^{13}$ pairwise comparison space down to a computationally tractable candidate set without sacrificing true matches:

- **Blocking Keys & Scheme**: 
  - Records are first partitioned strictly by `country` (matches across different countries are virtually non-existent and forbidden in practice).
  - Business name and address are cleaned, transliterated to ASCII, expanded for legal suffixes and street terms, and concatenated into a unified normalized representation.
  - A sublinear TF-IDF vectorizer extracts character n-grams ($n \in [3, 5]$) with a vocabulary capped at 100,000 features.
  - Pairwise similarity is computed using `awesome_cossim_topn`, retrieving the top-$N=20$ candidates per Source 1 entity above a cosine threshold of $\tau = 0.25$.
  - Large Source 2/Source 3 sets are processed in memory-safe chunks of 1,000,000 records.
- **Candidate Pairs Generated**: Average of ~15.2 candidates per Source 1 entity, yielding an effective reduction ratio of $> 99.9998\%$ of the Cartesian product space.
- **Ensuring True Matches Were Not Lost**: By employing character-level n-grams rather than strict word tokens or phonetic keys (such as Soundex/Metaphone), our blocking remains robust to severe typographical errors, partial name matches, and word-order transpositions. Validation testing proved that this blocking pass captures **95.58% of all true ground-truth matches** (blocking recall ceiling).

---

## 4. Matching Model

### Features Used (23 Pairwise Similarity Signals)
- **Name Features (11)**:
  - *Jaro-Winkler Similarity*: Captures typographical prefix similarities.
  - *Levenshtein Distance Ratio*: Global edit distance normalized by string length.
  - *Token Sort Ratio & Token Set Ratio*: Invariant to word reordering and subset containment.
  - *Partial String Ratio*: Accurately matches substring names and acronym expansions.
  - *Character 3-gram & 4-gram Jaccard Similarity*: Substring overlap insensitive to word boundaries.
  - *Token Overlap*: Jaccard overlap on whitespace-delimited word tokens.
  - *Common Prefix Length*: Exact character length of shared start tokens.
  - *Length Ratio & Absolute Length Difference*: Quantifies disparity in entity name descriptions.
- **Address Features (7)**:
  - *Address Levenshtein Ratio*: Global address character similarity.
  - *Address Token Sort & Token Set Ratio*: Order-independent address token overlap.
  - *Address Token Overlap Fraction*: Proportion of shared address words.
  - *Address Numeric Match Fraction*: Strict overlap of numeric digits (PIN codes, house numbers, street numbers), which acts as a powerful discriminator against distinct locations.
  - *Address Length Ratio & Missing Address Indicator*: Penalizes uninformative or absent address fields.
- **Composite & Cross-Field Features (5)**:
  - *TF-IDF Cosine Blocking Score*: Upstream vector similarity from the blocking phase.
  - *Name Average & Max Similarity*: Aggregations across top name metrics.
  - *Address Average Similarity*: Composite metric across address signals.
  - *Combined Similarity Score*: Weighted linear combination ($0.6 \times \text{Name} + 0.4 \times \text{Address}$).

### Model Architecture
- **Model Type**: LightGBM Gradient Boosted Decision Tree Classifier (`LGBMClassifier`).
  - `n_estimators`: 1,000 (with early stopping at 50 rounds)
  - `max_depth`: 8
  - `num_leaves`: 63
  - `learning_rate`: 0.05
  - `subsample`: 0.8, `colsample_bytree`: 0.8
  - `scale_pos_weight`: Balanced dynamically based on positive-to-negative candidate pair ratio (~3.85)
  - `reg_alpha`: 0.1, `reg_lambda`: 1.0
  - License: MIT License (< 10 MB artifact size, well under 8 Billion parameters).

### Threshold Selection Method
Because the competition evaluates Macro $F_{0.5}$—where Precision is weighted 2× more heavily than Recall—a standard 0.5 decision threshold results in excessive false merges that drastically penalize the score. We performed an empirical grid search from $t = 0.10$ to $0.96$ with step $0.02$ on an entity-isolated validation holdout:
- At $t = 0.50$: $F_{0.5} \approx 0.864$ (excess false positives)
- At $t = 0.80$: $F_{0.5} = 0.9461$
- At $t = 0.88$: **$F_{0.5} = 0.9474$ (Optimal)**
- Above $t = 0.92$: $F_{0.5} = 0.9458$ (precision saturates, recall begins dropping)

The selected threshold $t = 0.88$ guarantees that only pairs with very high confidence are merged, protecting singletons and distinct branch locations.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score (Validation)**: **0.9474**
  - Validation Precision: **0.962**
  - Validation Recall: **0.894**
  - Blocking Recall Ceiling: **0.9558**
- **Common False Positives (Wrong Merges)**:
  - *Franchise Branches & Chain Stores*: Identical business names (e.g., "Subway", "State Bank of India") operating in the same city but at different street locations where addresses are partially missing or truncated.
  - *Corporate Conglomerates*: Sister enterprises sharing primary name stems (e.g., "Tata Steel Limited" vs. "Tata Motors Limited") located in common industrial parks.
- **Common False Negatives (Missed Matches)**:
  - *Extreme Phonetic / Transliteration Distortions*: Non-English regional names transliterated into Latin characters with divergent spelling variants that fell below the TF-IDF blocking threshold.
  - *Complete Address Absence*: Records where address fields in Source 2 or Source 3 were completely blank or contained only a generic country label, withholding sufficient discriminatory information.

---

## 6. Conclusion

Our solution couples a memory-efficient character n-gram blocking engine with a 23-feature gradient boosted tree matching model. By isolating entities at the validation split and optimizing thresholding directly for the precision-focused macro $F_{0.5}$ metric, we achieve a robust 0.9474 validation score. The pipeline is lightweight (< 10 MB), strictly adheres to open-source MIT licensing, utilizes zero external lookups, and cleanly scales to multi-million record test sets.

---

## Appendix

### A. Code Artefacts
The complete runnable code is organized under `code/business_entity_resolution/`:
```text
code/business_entity_resolution/
├── README.md               # End-to-end reproduction guide
├── requirements.txt        # Pinned dependencies (lightgbm, rapidfuzz, unidecode, etc.)
└── src/
    ├── __init__.py
    ├── preprocess.py       # Text cleaning & normalization routines
    ├── blocking.py         # Sublinear TF-IDF character n-gram blocking
    ├── features.py         # 23-dimensional pairwise feature extraction
    ├── model.py            # LightGBM training, evaluation, threshold calibration
    ├── pipeline.py         # Full pipeline CLI (train, predict, sharded inference)
    └── model_artifacts/    # Trained model binary & metadata
```
To reproduce the outputs end-to-end:
```bash
# 1. Train model & calibrate threshold
python code/business_entity_resolution/src/pipeline.py --mode train

# 2. Run inference on test dataset
python code/business_entity_resolution/src/pipeline.py --mode predict

# 3. If running distributed shards across multiple machines:
python utils/merge_shards.py

# 4. Validate output format
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### B. Additional Results

#### Top-10 Most Important Features in LightGBM Matching Model
| Rank | Feature | Importance (Split Count) | Category | Description |
|:---:|:---|:---:|:---:|:---|
| 1 | `tfidf_score` | 2,666 | Composite | Upstream cosine similarity from char n-gram blocking |
| 2 | `addr_token_set` | 1,950 | Address | Token set similarity between addresses |
| 3 | `addr_len_ratio` | 1,942 | Address | Ratio of shorter to longer address length |
| 4 | `addr_token_overlap` | 1,614 | Address | Fraction of shared address tokens |
| 5 | `combined_score` | 1,536 | Composite | Weighted name + address similarity combination |
| 6 | `addr_token_sort` | 1,498 | Address | Order-invariant token match on addresses |
| 7 | `name_levenshtein` | 1,485 | Name | Normalized edit distance between business names |
| 8 | `name_len_diff` | 1,392 | Name | Absolute character length difference in names |
| 9 | `name_len_ratio` | 1,334 | Name | Length ratio of business names |
| 10 | `name_partial` | 1,325 | Name | Substring / partial matching ratio |

#### Threshold Calibration Sweep
| Threshold ($t$) | Macro $F_{0.5}$ | Precision | Recall |
|:---:|:---:|:---:|:---:|
| 0.76 | 0.9438 | 0.949 | 0.922 |
| 0.80 | 0.9461 | 0.954 | 0.913 |
| 0.84 | 0.9472 | 0.958 | 0.903 |
| 0.86 | 0.9474 | 0.960 | 0.899 |
| **0.88 (Optimal)** | **0.9474** | **0.962** | **0.894** |
| 0.90 | 0.9471 | 0.965 | 0.882 |
| 0.92 | 0.9458 | 0.968 | 0.867 |
