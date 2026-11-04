# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Package complete PIERE-2026 revision bundle for review and analysis."""

import os
from pathlib import Path
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results" / "revision"
OUT_ZIP = REPO_ROOT / "revision_bundle_complete.zip"


def create_complete_bundle():
    """Verify all revision deliverables and package into a single zip bundle."""
    print("=" * 80)
    print("  PACKAGING COMPLETE PIERE-2026 REVISION BUNDLE")
    print(f"  Repository root : {REPO_ROOT}")
    print(f"  Results dir     : {RESULTS_DIR}")
    print(f"  Output bundle   : {OUT_ZIP}")
    print("=" * 80)

    # Core required deliverables
    required_files = [
        ("SUMMARY.md", RESULTS_DIR / "SUMMARY.md"),
        ("metrics_all.csv", RESULTS_DIR / "metrics_all.csv"),
        ("stats_val.csv", RESULTS_DIR / "stats_val.csv"),
        ("stats_test.csv", RESULTS_DIR / "stats_test.csv"),
        ("schedule_ratios_val.csv", RESULTS_DIR / "schedule_ratios_val.csv"),
        ("schedule_ratios_test.csv", RESULTS_DIR / "schedule_ratios_test.csv"),
        ("corr_all.csv", RESULTS_DIR / "corr_all.csv"),
        ("illum.csv", RESULTS_DIR / "illum.csv"),
        ("optimizer_audit.csv", RESULTS_DIR / "info" / "optimizer_audit.csv"),
        ("split_audit.csv", RESULTS_DIR / "info" / "split_audit.csv"),
        ("exdark_splits.csv", RESULTS_DIR / "info" / "exdark_splits.csv"),
        ("hyperparams_summary.csv", RESULTS_DIR / "info" / "hyperparams_summary.csv"),
        ("val_curves_summary.csv", RESULTS_DIR / "info" / "val_curves_summary.csv"),
        ("val_curves_map50.csv", RESULTS_DIR / "info" / "val_curves_map50.csv"),
        ("methodology_ru.md", RESULTS_DIR / "info" / "methodology_ru.md"),
        ("fig1_mechanism_capacity.png", RESULTS_DIR / "figures" / "fig1_mechanism_capacity.png"),
        ("fig2_schedule_duration.png", RESULTS_DIR / "figures" / "fig2_schedule_duration.png"),
        ("REVISION_NOTES.md", REPO_ROOT / "REVISION_NOTES.md"),
        ("README.md", REPO_ROOT / "scripts" / "revision" / "README.md"),
    ]

    print("\n[VERIFICATION OF REQUIRED DELIVERABLES]:")
    all_found = True
    for label, path in required_files:
        status = "[OK]" if path.is_file() else "[MISSING]"
        size = f"{round(path.stat().st_size / 1024, 1)} KB" if path.is_file() else "---"
        print(f"  {status:<10} {label:<32} {size:>10}")
        if not path.is_file():
            all_found = False

    if not all_found:
        print("\n[WARN] Some deliverables are missing! Packaging all available files...")

    # Build zip
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()

    file_count = 0
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Add entire results/revision (excluding cache/smoke)
        if RESULTS_DIR.is_dir():
            for root, dirs, files in os.walk(RESULTS_DIR):
                if "smoke" in root or "__pycache__" in root:
                    continue
                for f in files:
                    if f.endswith(".zip"):
                        continue
                    full_p = Path(root) / f
                    arc_p = Path("results/revision") / full_p.relative_to(RESULTS_DIR)
                    zf.write(full_p, arcname=str(arc_p))
                    file_count += 1

        # 2. Add REVISION_NOTES.md
        rev_notes = REPO_ROOT / "REVISION_NOTES.md"
        if rev_notes.is_file():
            zf.write(rev_notes, arcname="REVISION_NOTES.md")
            file_count += 1

        # 3. Add scripts/revision
        scripts_dir = REPO_ROOT / "scripts" / "revision"
        if scripts_dir.is_dir():
            for root, dirs, files in os.walk(scripts_dir):
                if "__pycache__" in root:
                    continue
                for f in files:
                    full_p = Path(root) / f
                    arc_p = Path("scripts/revision") / full_p.relative_to(scripts_dir)
                    zf.write(full_p, arcname=str(arc_p))
                    file_count += 1

    zip_size_mb = round(OUT_ZIP.stat().st_size / (1024 * 1024), 2)
    print("\n" + "=" * 80)
    print("  ARCHIVE SUCCESSFULLY CREATED!")
    print(f"  Archive location : {OUT_ZIP.resolve()}")
    print(f"  Total files      : {file_count}")
    print(f"  Archive size     : {zip_size_mb} MB")
    print("=" * 80 + "\n")
    return OUT_ZIP


if __name__ == "__main__":
    create_complete_bundle()
