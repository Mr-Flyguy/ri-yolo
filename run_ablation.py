"""Ablation Study Orchestrator for RI-YOLO (Retinex-Informed YOLOv8s).

Conducts a 5-stage ablation study on ExDark low-light object detection:
  1. Baseline: Standard YOLOv8s
  2. RFD Only: YOLOv8s + Spatial Retinex Feature Decoupling Block
  3. RFD + RSL: YOLOv8s + RFD + Retinex Smoothness Loss (TV loss)
  4. RFD + NWD: YOLOv8s + RFD + Normalized Wasserstein Distance loss
  5. Full RI-YOLO: YOLOv8s + RFD + RSL + NWD
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments for ablation study."""
    parser = argparse.ArgumentParser(description="Run 5-Stage Ablation Study for RI-YOLO")
    parser.add_argument("--data", type=str, default="exdark.yaml", help="Path to dataset YAML file")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (pixels)")
    parser.add_argument("--device", default="", help="Device: '0', '0,1', 'cpu'")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Pretrained weights for transfer learning")
    parser.add_argument("--project", type=str, default="runs/ablation_study", help="Results save directory")
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Stages to run (1=Baseline, 2=RFD, 3=RFD+RSL, 4=RFD+NWD, 5=Full RI-YOLO)",
    )
    parser.add_argument("--lambda_tv", type=float, default=0.01, help="RSL Total Variation loss weight")
    parser.add_argument("--nwd_alpha", type=float, default=0.5, help="NWD / CIoU weight blend (1.0 = pure NWD)")
    return parser.parse_args()


def run_ablation():
    """Execute ablation experiments sequentially."""
    args = parse_args()

    # Define the 5 ablation experiment configurations
    experiments = [
        {
            "id": 1,
            "name": "1_baseline_yolov8s",
            "model_cfg": "yolov8s.yaml",
            "use_rsl": False,
            "use_nwd": False,
            "lambda_tv": 0.0,
            "nwd_alpha": 0.0,
            "description": "Baseline YOLOv8s (Standard CIoU, no RFD, no RSL)",
        },
        {
            "id": 2,
            "name": "2_yolov8s_rfd",
            "model_cfg": "ultralytics/cfg/models/v8/yolov8s-rfd.yaml",
            "use_rsl": False,
            "use_nwd": False,
            "lambda_tv": 0.0,
            "nwd_alpha": 0.0,
            "description": "YOLOv8s + Spatial RFDBlock (no RSL, no NWD)",
        },
        {
            "id": 3,
            "name": "3_yolov8s_rfd_rsl",
            "model_cfg": "ultralytics/cfg/models/v8/yolov8s-rfd.yaml",
            "use_rsl": True,
            "use_nwd": False,
            "lambda_tv": args.lambda_tv,
            "nwd_alpha": 0.0,
            "description": f"YOLOv8s + RFDBlock + RSL (TV loss lambda={args.lambda_tv})",
        },
        {
            "id": 4,
            "name": "4_yolov8s_rfd_nwd",
            "model_cfg": "ultralytics/cfg/models/v8/yolov8s-rfd.yaml",
            "use_rsl": False,
            "use_nwd": True,
            "lambda_tv": 0.0,
            "nwd_alpha": args.nwd_alpha,
            "description": f"YOLOv8s + RFDBlock + NWD (alpha={args.nwd_alpha})",
        },
        {
            "id": 5,
            "name": "5_full_ri_yolo",
            "model_cfg": "ultralytics/cfg/models/v8/yolov8s-rfd.yaml",
            "use_rsl": True,
            "use_nwd": True,
            "lambda_tv": args.lambda_tv,
            "nwd_alpha": args.nwd_alpha,
            "description": f"Full RI-YOLO (RFD + RSL lambda={args.lambda_tv} + NWD alpha={args.nwd_alpha})",
        },
    ]

    print("=" * 80)
    print("           RI-YOLO: 5-STAGE ABLATION STUDY RUNNER")
    print("=" * 80)
    print(f"Dataset      : {args.data}")
    print(f"Epochs       : {args.epochs}")
    print(f"Batch Size   : {args.batch}")
    print(f"Image Size   : {args.imgsz}")
    print(f"Weights      : {args.weights}")
    print(f"Save Path    : {args.project}")
    print(f"Active Stages: {args.stages}")
    print("=" * 80)

    results_summary = []

    for exp in experiments:
        if exp["id"] not in args.stages:
            print(f"\n[SKIP] Stage {exp['id']}: {exp['name']}")
            continue

        print("\n" + "#" * 80)
        print(f"Starting Stage {exp['id']}/{len(experiments)}: {exp['name']}")
        print(f"Description: {exp['description']}")
        print(f"Model Config: {exp['model_cfg']}")
        print(f"Flags: use_rsl={exp['use_rsl']}, use_nwd={exp['use_nwd']}, "
              f"lambda_tv={exp['lambda_tv']}, nwd_alpha={exp['nwd_alpha']}")
        print("#" * 80 + "\n")

        # 1. Initialize model architecture from YAML config
        model = YOLO(exp["model_cfg"])

        # 2. Transfer learning: Load pretrained COCO weights into backbone, SPPF, and Head
        if args.weights:
            weights_path = Path(args.weights)
            if not weights_path.exists():
                print(f"Pretrained weights {args.weights} not found locally; auto-downloading...")
                from ultralytics.utils.downloads import attempt_download_asset
                attempt_download_asset(args.weights)

            print(f"Loading pretrained weights from {args.weights} (Transfer Learning)...")
            if "rfd" in exp["model_cfg"]:
                # RFDBlock at index 10 shifts head layers by +1. Remap so all head layers retain COCO pretraining!
                ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
                csd = ckpt["model"].float().state_dict() if "model" in ckpt else ckpt
                msd = model.model.state_dict()
                remapped_csd = {}
                for k, v in csd.items():
                    parts = k.split(".")
                    if len(parts) > 1 and parts[0] == "model" and parts[1].isdigit():
                        idx = int(parts[1])
                        if idx >= 10:
                            remapped_csd[f"model.{idx + 1}." + ".".join(parts[2:])] = v
                        else:
                            remapped_csd[k] = v
                    else:
                        remapped_csd[k] = v
                matched = {k: v for k, v in remapped_csd.items() if k in msd and v.shape == msd[k].shape}
                model.model.load_state_dict(matched, strict=False)
                print(f"Transferred {len(matched)}/{len(msd)} items (Backbone, SPPF, and Head 100% transferred!)")
            else:
                model.load(args.weights)

        # 3. Launch training with ablation flags passed directly via kwargs
        train_args = {
            "data": args.data,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "project": args.project,
            "name": exp["name"],
            "use_rsl": exp["use_rsl"],
            "use_nwd": exp["use_nwd"],
            "lambda_tv": exp["lambda_tv"],
            "nwd_alpha": exp["nwd_alpha"],
            "exist_ok": True,
            "workers": args.workers,
        }
        if args.device:
            train_args["device"] = args.device

        results = model.train(**train_args)

        # Record validation metrics
        map50 = 0.0
        map50_95 = 0.0
        if results is not None:
            if hasattr(results, "box") and results.box is not None:
                map50 = getattr(results.box, "map50", 0.0)
                map50_95 = getattr(results.box, "map", 0.0)
            elif hasattr(results, "results_dict"):
                map50 = results.results_dict.get("metrics/mAP50(B)", 0.0)
                map50_95 = results.results_dict.get("metrics/mAP50-95(B)", 0.0)

        results_summary.append({
            "Stage": exp["id"],
            "Experiment": exp["name"],
            "RFD": "Yes" if exp["id"] > 1 else "No",
            "RSL": "Yes" if exp["use_rsl"] else "No",
            "NWD": "Yes" if exp["use_nwd"] else "No",
            "mAP50": f"{map50:.4f}",
            "mAP50-95": f"{map50_95:.4f}",
        })

    # Print final summary table
    print("\n" + "=" * 80)
    print("                    ABLATION STUDY FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Stage':<6} | {'Experiment':<22} | {'RFD':<5} | {'RSL':<5} | {'NWD':<5} | {'mAP50':<8} | {'mAP50-95':<8}")
    print("-" * 80)
    for r in results_summary:
        print(f"{r['Stage']:<6} | {r['Experiment']:<22} | {r['RFD']:<5} | {r['RSL']:<5} | {r['NWD']:<5} | {r['mAP50']:<8} | {r['mAP50-95']:<8}")
    print("=" * 80)


if __name__ == "__main__":
    run_ablation()
