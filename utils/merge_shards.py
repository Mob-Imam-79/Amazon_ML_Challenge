#!/usr/bin/env python3
"""
utils/merge_shards.py — Pure Python standard library merger for sharded predictions.
Requires ZERO external packages (no pandas required). Runs lightning-fast.
"""

import argparse
import glob
import os
import re
import subprocess
import sys


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def merge_shards_stdlib(file_list, output_path, col_header):
    if not file_list:
        return 0

    print(f"\n  Merging {len(file_list)} file(s) into {output_path}:")
    for f in file_list:
        size_mb = os.path.getsize(f) / (1024 * 1024)
        print(f"    - {f} ({size_mb:.1f} MB)")

    data = {}
    for f in file_list:
        with open(f, 'r', encoding='utf-8') as fh:
            header = fh.readline()
            for line in fh:
                if not line.strip():
                    continue
                parts = line.rstrip('\r\n').split('\t')
                s1_id = parts[0].strip()
                val = parts[1].strip() if len(parts) > 1 else ""
                data[s1_id] = val

    print(f"  Sorting and writing {len(data):,} unique records...")
    with open(output_path, 'w', encoding='utf-8') as out_fh:
        out_fh.write(f"{col_header[0]}\t{col_header[1]}\n")
        for s1_id in sorted(data.keys()):
            out_fh.write(f"{s1_id}\t{data[s1_id]}\n")

    print(f"  ✅ Saved {output_path} ({len(data):,} rows)")
    return len(data)


def main():
    parser = argparse.ArgumentParser(description="Merge sharded predictions and candidate pairs.")
    parser.add_argument("--output-dir", default="output", help="Directory containing shard files")
    parser.add_argument("--test-dir", default="dataset/test", help="Directory with test data")
    args = parser.parse_args()

    matching_shards = sorted(glob.glob(os.path.join(args.output_dir, "matching_results_shard_*.tsv")), key=natural_sort_key)
    candidate_shards = sorted(glob.glob(os.path.join(args.output_dir, "candidate_pairs_shard_*.tsv")), key=natural_sort_key)

    matching_out = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_out = os.path.join(args.output_dir, "candidate_pairs.tsv")

    print("=" * 65)
    print("  Entity Resolution — Merge Shards & Validate (Stdlib)")
    print("=" * 65)

    s1_test_path = os.path.join(args.test_dir, "test_source1.tsv")
    total_expected = 1732544
    if os.path.exists(s1_test_path):
        with open(s1_test_path, 'r', encoding='utf-8') as f:
            total_expected = sum(1 for _ in f) - 1
    print(f"  Total required test S1 entities: {total_expected:,}")

    # Merge matching shards
    matching_dict = {}
    if matching_shards:
        print(f"\n  Found {len(matching_shards)} matching shard(s):")
        for f in matching_shards:
            print(f"    • {f}")
            with open(f, 'r', encoding='utf-8') as fh:
                next(fh)
                for line in fh:
                    if not line.strip():
                        continue
                    parts = line.rstrip('\r\n').split('\t')
                    matching_dict[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""
    else:
        print("  No matching_results_shard_*.tsv found.")

    # Merge candidate shards
    candidate_dict = {}
    if candidate_shards:
        print(f"\n  Found {len(candidate_shards)} candidate shard(s):")
        for f in candidate_shards:
            print(f"    • {f}")
            with open(f, 'r', encoding='utf-8') as fh:
                next(fh)
                for line in fh:
                    if not line.strip():
                        continue
                    parts = line.rstrip('\r\n').split('\t')
                    candidate_dict[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""

    # Write full matching_results.tsv and candidate_pairs.tsv ensuring all 1.73M entities exist
    print(f"\n  Writing full {total_expected:,} rows to {matching_out} and {candidate_out}...")
    with open(s1_test_path, 'r', encoding='utf-8') as fs1, \
         open(matching_out, 'w', encoding='utf-8') as fm, \
         open(candidate_out, 'w', encoding='utf-8') as fc:

        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

        next(fs1)
        count = 0
        for line in fs1:
            if not line.strip():
                continue
            s1_id = line.split('\t', 1)[0].strip()
            m = matching_dict.get(s1_id, "")
            c = candidate_dict.get(s1_id, m)
            fm.write(f"{s1_id}\t{m}\n")
            fc.write(f"{s1_id}\t{c}\n")
            count += 1

    print(f"  ✅ Wrote all {count:,} entities.")
    print(f"     Matched records: {len([v for v in matching_dict.values() if v]):,}")
    print(f"     Singletons / Blank: {count - len([v for v in matching_dict.values() if v]):,}")

    # Validate
    print("\n" + "=" * 65)
    print("  Running Official Submission Validator")
    print("=" * 65)
    validator_path = os.path.join("utils", "validate_submission.py")
    if os.path.exists(validator_path):
        subprocess.run([
            sys.executable,
            validator_path,
            "--matching", matching_out,
            "--candidate", candidate_out,
            "--test-dir", args.test_dir
        ])


if __name__ == "__main__":
    main()
