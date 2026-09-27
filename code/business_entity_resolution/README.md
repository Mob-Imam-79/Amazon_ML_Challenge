# Business Entity Resolution Pipeline

This repository contains an end-to-end, high-performance Machine Learning pipeline for the **Amazon ML Challenge 2026: Business Entity Resolution**.

Given business records across three independent data sources (`Source 1`, `Source 2`, and `Source 3`) containing noisy names, varied addresses, and missing attributes, this pipeline deduplicates and resolves all matching records from Sources 2 and 3 for every reference record in Source 1.

---

## 1. Directory Structure

```text
code/business_entity_resolution/
├── README.md                 # This reproduction guide
├── requirements.txt          # Pinned Python package dependencies
└── src/
    ├── __init__.py
    ├── preprocess.py         # Text cleaning, normalization, transliteration, address tokens
    ├── blocking.py           # Country-partitioned TF-IDF n-gram candidate generation
    ├── features.py           # 23 pairwise name, address, and combined similarity features
    ├── model.py              # LightGBM classifier, entity-level validation, F_0.5 threshold sweep
    ├── pipeline.py           # Main CLI driver supporting train, predict, and distributed sharding
    └── model_artifacts/      # Saved trained model (model.txt, model_meta.json)
```

---

## 2. Environment Setup & Requirements

The pipeline requires **Python 3.8+** (tested on Python 3.11 & 3.14 on macOS Apple Silicon and Linux/Windows).

### Installation

From the repository root (`student_resource/`):

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install --upgrade pip
pip install -r code/business_entity_resolution/requirements.txt
```

### Dependencies
- `pandas >= 2.0`
- `numpy >= 1.24`
- `scikit-learn >= 1.3`
- `lightgbm >= 4.0`
- `rapidfuzz >= 3.0`
- `unidecode >= 1.3`
- `sparse_dot_topn >= 1.1`
- `jellyfish >= 1.0`

---

## 3. Data Layout

The pipeline expects data organized as follows:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All files must be **tab-separated** (`.tsv`), UTF-8 encoded.

---

## 4. Pipeline Architecture

1. **Preprocessing (`src/preprocess.py`)**:
   - Strips noise tokens, URLs, emails, and collapses whitespace.
   - Standardizes legal suffixes (e.g. `pvt ltd` → `private limited`, `inc` → `incorporated`, `corp` → `corporation`).
   - Normalizes address abbreviations (`rd` → `road`, `st` → `street`, `ste` → `suite`, `fl` → `floor`, etc.).
   - Converts Indian & US state abbreviations and performs Unicode transliteration to ASCII using `unidecode`.
   - Extracts numeric components (PIN codes, house numbers) separately for strict numeric consistency checking.

2. **Candidate Generation / Blocking (`src/blocking.py`)**:
   - Partitioned by `country` (strictly separating `US`, `India`, and `France`).
   - Fits sublinear TF-IDF vectorizer over character n-grams (range: 3–5, vocabulary cap: 100,000).
   - Computes sparse top-$N$ cosine similarity using `awesome_cossim_topn` ($N=20$, threshold $\ge 0.25$).
   - Implements chunked matrix processing to maintain memory safety on machines with $\le 16\text{ GB}$ RAM.
   - Achieves **95.58% recall ceiling** while reducing candidate search space from $1.7\times 10^7$ down to ~15 pairs per entity.

3. **Feature Engineering (`src/features.py`)**:
   - Calculates 23 pairwise features across multiple string distance algorithms:
     - **Name Similarity (11)**: Jaro-Winkler, Levenshtein ratio, Token Sort ratio, Token Set ratio, Partial ratio, 3-gram & 4-gram Jaccard, Token Overlap, Common Prefix Length, Length Ratio, Length Difference.
     - **Address Similarity (7)**: Levenshtein ratio, Token Sort ratio, Token Set ratio, Token Overlap, Numeric Token Match fraction, Length Ratio, Missing Address Indicator.
     - **Composite Features (5)**: TF-IDF cosine score, Name Average similarity, Name Max similarity, Address Average similarity, and Weighted Composite Score ($0.6 \times \text{Name} + 0.4 \times \text{Address}$).

4. **Classification & Threshold Optimization (`src/model.py`)**:
   - Model: **LightGBM Classifier** (`n_estimators=1000`, `max_depth=8`, `num_leaves=63`, `learning_rate=0.05`, `scale_pos_weight` tuned for class imbalance).
   - **Entity-Level Validation Split**: Partitioned strictly by `source1_entity_id` (not random pairs) to prevent leakage of entity context across train and validation folds.
   - **Threshold Sweep**: Grid search maximizing the competition metric:
     $$\text{Macro } F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
     evaluated on all validation entities, including singletons ($F_{0.5} = 1.0$ for correct empty prediction, $0.0$ for false merge).
   - Optimal threshold identified: **$t = 0.88$**, achieving **$0.9474$** validation Macro $F_{0.5}$.

5. **Inference & Singleton Handling (`src/pipeline.py`)**:
   - Scores candidate pairs using the trained model and decision threshold.
   - Aggregates predicted IDs per Source 1 entity.
   - Guarantees complete coverage: any Source 1 entity with zero candidates or no candidate above threshold is recorded as an empty singleton string, ensuring every test S1 entity has exactly one row.

---

## 5. End-to-End Reproduction Steps

### Step 1: Train the Model

To train the model on the training dataset and tune the decision threshold:

```bash
# Full training on train data
python code/business_entity_resolution/src/pipeline.py --mode train

# Or run a fast prototype with a sample of 20,000 entities
python code/business_entity_resolution/src/pipeline.py --mode train --sample 20000
```
This saves:
- `code/business_entity_resolution/src/model_artifacts/model.txt`
- `code/business_entity_resolution/src/model_artifacts/model_meta.json`

### Step 2: Generate Predictions on Test Data

#### Option A: Single Machine Full Run
```bash
python code/business_entity_resolution/src/pipeline.py --mode predict
```
This directly outputs:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

#### Option B: Distributed / Sharded Inference (Multi-Machine)
For large test sets, the pipeline supports sharding:
```bash
# Machine 1 (Shard 1 of 4):
python code/business_entity_resolution/src/pipeline.py --mode predict --shard 1/4

# Machine 2 (Shard 2 of 4):
python code/business_entity_resolution/src/pipeline.py --mode predict --shard 2/4

# Machine 3 (Shard 3 of 4):
python code/business_entity_resolution/src/pipeline.py --mode predict --shard 3/4

# Machine 4 (Shard 4 of 4):
python code/business_entity_resolution/src/pipeline.py --mode predict --shard 4/4
```
Each shard produces:
- `output/matching_results_shard_X_of_4.tsv`
- `output/candidate_pairs_shard_X_of_4.tsv`

### Step 3: Merge Shards
When all shards have completed, place all shard files in `output/` and merge them safely:
```bash
python utils/merge_shards.py
```
This automatically:
- Concatenates matching results and candidate pairs without duplicate headers.
- Sorts rows by `source1_entity_id`.
- Verifies that all 1,732,544 test S1 entities are present.
- Generates `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- Runs the validation script to verify submission readiness.

### Step 4: Validate Outputs
Run the official challenge validator:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
A clean run exits with `0` and prints:
```text
PASS — no blocking issues found. Safe to submit.
```

---

## 6. Constraints & Compliance

- **Model Parameters**: LightGBM tree-based ensemble (< 10 MB, well under the 8 Billion parameter ceiling).
- **License**: MIT License (LightGBM, scikit-learn, rapidfuzz).
- **Fair Play**: 100% self-contained logic; strictly **no external APIs, geocoders, or online databases** were used.
