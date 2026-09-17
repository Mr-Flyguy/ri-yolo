# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Stage F.6: Artifact Reproducibility Verifier for Paper C2.

Verifies bit-for-bit reproducibility of:
- tables/C2_rsl_curves.csv (via scripts/export_c2_rsl_curves.py)
- tables/C2_table1.csv (via scripts/build_c2_stage_e.py)
- tables/C2_branch_scale.csv (via scripts/build_c2_stage_e.py)
- tables/C2_gamma_curves.csv (via scripts/build_c2_stage_e.py)

Steps:
1. Back up existing files to *_before.csv
2. Execute generator scripts
3. Perform exact line-by-line diff and report verbatim comparison
"""

import difflib
import os
from pathlib import Path
import shutil
import subprocess
import sys


FILES_TO_CHECK = [
    ("tables/C2_rsl_curves.csv", "python scripts/export_c2_rsl_curves.py"),
    ("tables/C2_table1.csv", "python scripts/build_c2_stage_e.py"),
    ("tables/C2_branch_scale.csv", "python scripts/build_c2_stage_e.py"),
    ("tables/C2_gamma_curves.csv", "python scripts/build_c2_stage_e.py"),
]


def compare_files(orig_p: Path, before_p: Path):
    print("\n" + "=" * 80)
    print(f"  DIFF CHECK: {orig_p} vs {before_p}")
    print("=" * 80)

    if not orig_p.exists():
        print(f"[ERROR] Generated file {orig_p} DOES NOT EXIST.")
        return False
    if not before_p.exists():
        print(f"[ERROR] Backup file {before_p} DOES NOT EXIST.")
        return False

    with open(before_p, "r", encoding="utf-8") as f1, open(orig_p, "r", encoding="utf-8") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    diff = list(difflib.unified_diff(lines1, lines2, fromfile=str(before_p), tofile=str(orig_p), lineterm=""))
    if not diff:
        print(f"[VERIFIED] {orig_p}: EXACT MATCH (100% reproducible, 0 differences, {len(lines2)} lines).")
        return True
    else:
        print(f"[DIVERGENCE DETECTED] {orig_p} differs from {before_p}:")
        for line in diff:
            print(line)
        return False


def main():
    print("=" * 80)
    print("  RI-YOLO: ARTIFACT REPRODUCIBILITY VERIFICATION (Stage F.6)")
    print("=" * 80)

    # 1. Backup files
    backups = {}
    for orig_str, _ in FILES_TO_CHECK:
        orig = Path(orig_str)
        before = orig.parent / f"{orig.stem}_before{orig.suffix}"
        if orig.exists():
            shutil.copyfile(orig, before)
            backups[orig_str] = before
            print(f"[BACKUP] Copied {orig} -> {before}")
        else:
            print(f"[WARN] Original {orig} does not exist prior to run.")

    # 2. Run generators
    executed_commands = set()
    for _, cmd in FILES_TO_CHECK:
        if cmd not in executed_commands:
            print(f"\n[EXEC] Running generator: {cmd}")
            ret = subprocess.run(cmd, shell=True, text=True, capture_output=True)
            print(ret.stdout)
            if ret.stderr:
                print("[STDERR]", ret.stderr)
            executed_commands.add(cmd)

    # 3. Compare
    all_ok = True
    for orig_str, _ in FILES_TO_CHECK:
        orig = Path(orig_str)
        before = backups.get(orig_str)
        if before:
            match = compare_files(orig, before)
            if not match:
                all_ok = False

    print("\n" + "=" * 80)
    if all_ok:
        print("  REPRODUCIBILITY RESULT: ALL ARTIFACTS REPRODUCE BIT-FOR-BIT")
    else:
        print("  REPRODUCIBILITY RESULT: DIVERGENCES FOUND (SEE DETAILS ABOVE)")
    print("=" * 80)


if __name__ == "__main__":
    main()
