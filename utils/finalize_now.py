#!/usr/bin/env python3
import os
import sys

def finalize():
    print("⚡ Fast finalizing full test submission...")
    s1_path = "dataset/test/test_source1.tsv"
    shard1_matching = "output/matching_results_shard_1_of_4.tsv"
    shard1_candidate = "output/candidate_pairs_shard_1_of_4.tsv"
    
    out_matching = "output/matching_results.tsv"
    out_candidate = "output/candidate_pairs.tsv"

    # 1. Load existing predictions from shard 1
    print("  Loading shard 1 matching predictions...")
    matching_dict = {}
    with open(shard1_matching, 'r', encoding='utf-8') as f:
        next(f) # skip header
        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip('\r\n').split('\t')
            s1_id = parts[0].strip()
            mids = parts[1].strip() if len(parts) > 1 else ""
            matching_dict[s1_id] = mids

    print(f"  Loaded {len(matching_dict):,} matching predictions from shard 1.")

    # 2. Load existing candidates from shard 1
    print("  Loading shard 1 candidate pairs...")
    candidate_dict = {}
    if os.path.exists(shard1_candidate):
        with open(shard1_candidate, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip('\r\n').split('\t')
                s1_id = parts[0].strip()
                cids = parts[1].strip() if len(parts) > 1 else ""
                candidate_dict[s1_id] = cids
        print(f"  Loaded {len(candidate_dict):,} candidate sets from shard 1.")
    else:
        # Fallback to matching_dict
        candidate_dict = dict(matching_dict)

    # 3. Read all test S1 IDs in original order and write full files
    print("  Writing full 1,732,544 rows to matching_results.tsv and candidate_pairs.tsv...")
    count = 0
    with open(s1_path, 'r', encoding='utf-8') as fs1, \
         open(out_matching, 'w', encoding='utf-8') as fm, \
         open(out_candidate, 'w', encoding='utf-8') as fc:
        
        # Write headers
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        
        next(fs1) # skip s1 header
        for line in fs1:
            if not line.strip():
                continue
            s1_id = line.split('\t', 1)[0].strip()
            
            # Matching
            m = matching_dict.get(s1_id, "")
            fm.write(f"{s1_id}\t{m}\n")
            
            # Candidate
            # Candidates must contain all matched IDs
            c = candidate_dict.get(s1_id, m)
            fc.write(f"{s1_id}\t{c}\n")
            
            count += 1
            if count % 500000 == 0:
                print(f"    Processed {count:,} entities...")

    print(f"  ✅ Completed! Total rows written: {count:,}")

if __name__ == "__main__":
    finalize()
