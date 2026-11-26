# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Phase 2 Table Aggregator: TV-loss Regularization and NWD Metric Formulations.

Aggregates results for Article C2 (VAK / RSCI):
- Table 1: E6' lambda_tv Regularization Sweep
- Table 2: E7' Bounding Box Metric Formulations
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Collect and generate Tables for Paper C2")
    parser.add_argument(
        "--summary",
        type=str,
        default="artifacts/phase2/summary.csv",
        help="Path to Phase 2 summary.csv",
    )
    parser.add_argument(
        "--out-t1",
        type=str,
        default="artifacts/phase2/C2_table1.csv",
        help="Output path for Table 1 (E6' TV-loss sweep)",
    )
    parser.add_argument(
        "--out-t2",
        type=str,
        default="artifacts/phase2/C2_table2.csv",
        help="Output path for Table 2 (E7' NWD variants)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    summary_file = Path(args.summary)

    if not summary_file.exists():
        print(f"[WARN] {summary_file} does not exist yet. Run Phase 2 experiments first.")
        return

    with open(summary_file) as f:
        rows = list(csv.DictReader(f))

    print(f"[INFO] Loaded {len(rows)} record(s) from {summary_file}.")

    # --- Table 1: E6' TV-loss sweep ---
    e6_rows = [r for r in rows if r.get("group") == "E6"]
    if e6_rows:
        out_t1 = Path(args.out_t1)
        out_t1.parent.mkdir(parents=True, exist_ok=True)
        with open(out_t1, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(e6_rows[0].keys()))
            writer.writeheader()
            writer.writerows(e6_rows)
        print(f"[SUCCESS] Table 1 written to {out_t1} ({len(e6_rows)} rows)")

    # --- Table 2: E7' NWD variants ---
    e7_rows = [r for r in rows if r.get("group") == "E7"]
    if e7_rows:
        out_t2 = Path(args.out_t2)
        out_t2.parent.mkdir(parents=True, exist_ok=True)
        with open(out_t2, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(e7_rows[0].keys()))
            writer.writeheader()
            writer.writerows(e7_rows)
        print(f"[SUCCESS] Table 2 written to {out_t2} ({len(e7_rows)} rows)")


if __name__ == "__main__":
    main()
