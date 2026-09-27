#!/usr/bin/env python3
"""
utils/create_submission_zip.py — Package final submission into the official structure.

Structure:
  <team_name>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md
"""

import argparse
import os
import sys
import zipfile


def build_zip(team_name, base_dir="."):
    zip_filename = f"{team_name}_submission.zip"
    zip_path = os.path.join(base_dir, zip_filename)

    output_dir = os.path.join(base_dir, "output")
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")

    code_ber_dir = os.path.join(base_dir, "code", "business_entity_resolution")
    doc_path = os.path.join(base_dir, "Documentation_template.md")

    # Validate essential files exist
    missing = []
    for p in [matching_tsv, candidate_tsv, code_ber_dir, doc_path]:
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        print(f"❌ Cannot create zip. The following required files/folders are missing:")
        for m in missing:
            print(f"   - {m}")
        return False

    print(f"📦 Packaging submission into: {zip_filename}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. output/ files
        zf.write(matching_tsv, arcname="output/matching_results.tsv")
        print(f"  + Added output/matching_results.tsv ({os.path.getsize(matching_tsv)/(1024*1024):.1f} MB)")
        zf.write(candidate_tsv, arcname="output/candidate_pairs.tsv")
        print(f"  + Added output/candidate_pairs.tsv ({os.path.getsize(candidate_tsv)/(1024*1024):.1f} MB)")

        # 2. code/business_entity_resolution files
        for root, dirs, files in os.walk(code_ber_dir):
            # Skip __pycache__ and git folders
            dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", ".pytest_cache")]
            for file in files:
                if file.endswith((".pyc", ".DS_Store")):
                    continue
                file_abs = os.path.join(root, file)
                rel_path = os.path.relpath(file_abs, base_dir)
                zf.write(file_abs, arcname=rel_path)
                print(f"  + Added {rel_path}")

        # 3. Documentation_template.md
        zf.write(doc_path, arcname="Documentation_template.md")
        print(f"  + Added Documentation_template.md")

    total_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\n✅ Successfully generated {zip_filename} ({total_mb:.1f} MB)!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create official submission zip package.")
    parser.add_argument("--team-name", default="Team_Mobashir", help="Your team name for the zip file")
    args = parser.parse_args()

    build_zip(args.team_name)
