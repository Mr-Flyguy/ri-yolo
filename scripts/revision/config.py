# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Configuration and shared definitions for the PIERE-2026 revision experiments."""

import hashlib
from pathlib import Path
import sys
import types
from typing import Dict, List, Optional, Union
import torch

TORCH_VERSION = str(getattr(torch, "__version__", "unknown"))
TORCHVISION_VERSION = "unavailable"
NMS_BACKEND = "torchvision"


def init_nms_backend(smoke: bool = False) -> str:
    """Validate torchvision availability and configure NMS backend.

    In production mode, raises RuntimeError if torchvision is unavailable or broken.
    In smoke mode, falls back to pure PyTorch NMS implementation and sets flag.
    """
    global NMS_BACKEND, TORCHVISION_VERSION
    try:
        import torchvision

        TORCHVISION_VERSION = str(getattr(torchvision, "__version__", "unknown"))
        test_boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32)
        test_scores = torch.tensor([0.9], dtype=torch.float32)
        _ = torchvision.ops.nms(test_boxes, test_scores, 0.5)
        NMS_BACKEND = "torchvision"
        return "torchvision"
    except Exception as e:
        if smoke:
            NMS_BACKEND = "pure_torch_smoke_fallback"

            def _pure_torch_nms(boxes, scores, iou_threshold):
                if boxes.numel() == 0:
                    return torch.empty((0,), dtype=torch.int64, device=boxes.device)
                x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                areas = (x2 - x1) * (y2 - y1)
                order = scores.argsort(descending=True)
                keep = []
                while order.numel() > 0:
                    i = order[0]
                    keep.append(i.item())
                    if order.numel() == 1:
                        break
                    xx1 = torch.max(x1[i], x1[order[1:]])
                    yy1 = torch.max(y1[i], y1[order[1:]])
                    xx2 = torch.min(x2[i], x2[order[1:]])
                    yy2 = torch.min(y2[i], y2[order[1:]])
                    w = torch.clamp(xx2 - xx1, min=0.0)
                    h = torch.clamp(yy2 - yy1, min=0.0)
                    inter = w * h
                    ovr = inter / (areas[i] + areas[order[1:]] - inter)
                    ids = (ovr <= iou_threshold).nonzero().squeeze(-1)
                    if ids.numel() == 0:
                        break
                    order = order[ids + 1]
                return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

            tv = types.ModuleType("torchvision")
            tv.ops = types.ModuleType("ops")
            tv.ops.nms = _pure_torch_nms
            sys.modules["torchvision"] = tv
            sys.modules["torchvision.ops"] = tv.ops
            print(
                f"[WARN] torchvision.ops.nms is unavailable ({e}). "
                "Active backend: pure_torch_smoke_fallback (ALLOWED EXCLUSIVELY FOR --smoke MODE)."
            )
            return "pure_torch_smoke_fallback"
        else:
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"FATAL ERROR: torchvision.ops.nms is missing or broken in production environment!\n"
                f"Detected: torch=={TORCH_VERSION}, torchvision=={TORCHVISION_VERSION} ({e})\n"
                "In accordance with revision specifications, production runs require verified torchvision.\n"
                "Fix torchvision on your GPU machine (e.g. 'pip install torchvision==0.20' for torch 2.5.x) "
                "or pass --smoke for CPU testing.\n"
                f"{'='*80}\n"
            )


def select_torch_device(device: Union[str, int, torch.device] = "0") -> torch.device:
    """Parse device string/int to a valid torch.device object."""
    if isinstance(device, torch.device):
        return device
    dev_str = str(device).strip()
    if dev_str.isdigit():
        return torch.device(f"cuda:{dev_str}" if torch.cuda.is_available() else "cpu")
    if dev_str.lower() in ("cuda", "gpu"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev_str.lower() == "cpu":
        return torch.device("cpu")
    try:
        return torch.device(dev_str)
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Base filesystem directories
REVISION_ROOT = Path(__file__).resolve().parent
REPO_ROOT = REVISION_ROOT.parent.parent
DEFAULT_DATA_YAML = REPO_ROOT / "exdark.yaml"
DIAG_IMAGE_LIST = REPO_ROOT / "artifacts" / "phase2" / "diag_image_list.txt"

# Results and artifacts directories
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "revision"
DEFAULT_SMOKE_DIR = REPO_ROOT / "results" / "revision_smoke"

# Evaluation protocol constants (strict reproducibility)
VAL_PROTOCOL_NAME = "Ultralytics DetectionValidator (faster-coco-eval)"
VAL_PARAMS_DICT = {
    "imgsz": 640,
    "batch": 16,
    "conf": 0.001,
    "iou": 0.6,
    "max_det": 300,
    "rect": False,
    "half": False,
}

# Module architecture passport constants
MODULE_ARCH_INFO = {
    "module_class": "RFDBlock",
    "control_class": "RFDBlockNoGate",
    "yaml_layer_index": 10,
    "input_layer": "SPPF (layer 9, P5/32)",
    "input_channels": 512,
    "output_channels": 512,
    "stride": 32,
    "map_resolution_at_640": (20, 20),
    "rfd_total_params": 2624513,
    "capctrl_total_params": 2624001,
    "estimator_params": 512,  # 512 * 1 * 1 Conv2d without bias
    "baseline_model_params_M": 11.167,
    "postsppf_model_params_M": 13.791,
    "capctrl_model_params_M": 13.790,
    "baseline_gflops": 28.81,
    "postsppf_gflops": 30.91,
}

# Historical runtime estimates on RTX 3070 Ti (seconds per epoch)
HISTORICAL_EPOCH_TIME_SEC = 25.0

# ---------------------------------------------------------------------------
# Experiment Catalog: Existing Checkpoints (32 runs from C1, C2, C3)
# ---------------------------------------------------------------------------
EXISTING_RUN_SPECS: List[Dict] = [
    # Baseline (4 seeds, 100 epochs)
    {
        "run": "e2p__baseline__s0",
        "family": "baseline",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "yolov8s.yaml",
        "candidates": [
            "runs/detect/runs/e2p__baseline__s0/weights/best.pt",
            "runs/e2p__baseline__s0/weights/best.pt",
        ],
        "md5": "6b4d63a61e64a1016fe8c99155f152ba",
    },
    {
        "run": "e2p__baseline__s1",
        "family": "baseline",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 1,
        "cfg": "yolov8s.yaml",
        "candidates": [
            "runs/detect/runs/e2p__baseline__s1/weights/best.pt",
            "runs/e2p__baseline__s1/weights/best.pt",
        ],
        "md5": "f9f720a1740f33b6d3a80ef43bdbc95d",
    },
    {
        "run": "e2p__baseline__s2",
        "family": "baseline",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 2,
        "cfg": "yolov8s.yaml",
        "candidates": [
            "runs/detect/runs/e2p__baseline__s2/weights/best.pt",
            "runs/e2p__baseline__s2/weights/best.pt",
        ],
        "md5": "8ffe393f63b8a5568198100986fffd1a",
    },
    {
        "run": "e2p__baseline__s3",
        "family": "baseline",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 3,
        "cfg": "yolov8s.yaml",
        "candidates": [
            "runs/detect/runs/e2p__baseline__s3/weights/best.pt",
            "runs/e2p__baseline__s3/weights/best.pt",
        ],
        "md5": "c58e406f4a49fb5feb502c9f73c2a28a",
    },
    # RFD Post-SPPF (4 seeds, 100 epochs, canonical lambda=0 / implicit)
    {
        "run": "e2p__postsppf__s0",
        "family": "postsppf",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e3p__pos-postsppf__s0/weights/best.pt",
            "runs/detect/runs/e2p__postsppf__s0/weights/best.pt",
            "runs/e3p__pos-postsppf__s0/weights/best.pt",
            "runs/e2p__postsppf__s0/weights/best.pt",
        ],
        "md5": "a5510ad1b6f3ea7f4702d3c63a0df016",
    },
    {
        "run": "e2p__postsppf__s1",
        "family": "postsppf",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 1,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e2p__postsppf__s1/weights/best.pt",
            "runs/e2p__postsppf__s1/weights/best.pt",
        ],
        "md5": "6cf304e6e96454f3e76c59b70da5cb52",
    },
    {
        "run": "e2p__postsppf__s2",
        "family": "postsppf",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 2,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e2p__postsppf__s2/weights/best.pt",
            "runs/e2p__postsppf__s2/weights/best.pt",
        ],
        "md5": "674032c468478b6f934b01aaf0694412",
    },
    {
        "run": "e2p__postsppf__s3",
        "family": "postsppf",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 3,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e2p__postsppf__s3/weights/best.pt",
            "runs/e2p__postsppf__s3/weights/best.pt",
        ],
        "md5": "8712af949baab0eef8305e962bca4f05",
    },
    # Capacity Control (3 seeds, 100 epochs, constant 0.5 gate)
    {
        "run": "e3p__capctrl__s0",
        "family": "capctrl",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-capctrl.yaml",
        "candidates": [
            "runs/detect/runs/e3p__capctrl__s0/weights/best.pt",
            "runs/e3p__capctrl__s0/weights/best.pt",
        ],
        "md5": "5921a4e2556d411b3e5b4b27f0ff0a18",
    },
    {
        "run": "e3p__capctrl__s1",
        "family": "capctrl",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 1,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-capctrl.yaml",
        "candidates": [
            "runs/detect/runs/e3p__capctrl__s1/weights/best.pt",
            "runs/e3p__capctrl__s1/weights/best.pt",
        ],
        "md5": "229d733ef8dd8ef07531712df7458aaf",
    },
    {
        "run": "e3p__capctrl__s2",
        "family": "capctrl",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 2,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-capctrl.yaml",
        "candidates": [
            "runs/detect/runs/e3p__capctrl__s2/weights/best.pt",
            "runs/e3p__capctrl__s2/weights/best.pt",
        ],
        "md5": "01deac86eb112a00be89a201536b172f",
    },
    # Schedule Ablations (Seed 0)
    {
        "run": "e5p__baseline-ep40__s0",
        "family": "baseline",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "yolov8s.yaml",
        "candidates": [
            "runs/detect/runs/e5p__baseline-ep40__s0/weights/best.pt",
            "runs/e5p__baseline-ep40__s0/weights/best.pt",
        ],
        "md5": "b983b42ed4b2431cb5d08d834ddd3868",
    },
    {
        "run": "e5p__baseline-ep60__s0",
        "family": "baseline",
        "epochs": 60,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "yolov8s.yaml",
        "candidates": [
            "runs/detect/runs/e5p__baseline-ep60__s0/weights/best.pt",
            "runs/e5p__baseline-ep60__s0/weights/best.pt",
        ],
        "md5": "e8c2f7f0f08d8c683143946ae0e7f2b2",
    },
    {
        "run": "e5p__postsppf-ep40__s0",
        "family": "postsppf",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e5p__postsppf-ep40__s0/weights/best.pt",
            "runs/e5p__postsppf-ep40__s0/weights/best.pt",
        ],
        "md5": "12265a1f25c6f44a5729e10c6ee7ec18",
    },
    {
        "run": "e5p__postsppf-ep60__s0",
        "family": "postsppf",
        "epochs": 60,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e5p__postsppf-ep60__s0/weights/best.pt",
            "runs/e5p__postsppf-ep60__s0/weights/best.pt",
        ],
        "md5": "409a806d12393b5eac2c32869dc43829",
    },
    # Mosaic Sensitivity Ablations (Seed 0)
    {
        "run": "e4p__mosaic-close0__s0",
        "family": "postsppf",
        "epochs": 100,
        "close_mosaic": 0,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e4p__mosaic-close0__s0/weights/best.pt",
            "runs/e4p__mosaic-close0__s0/weights/best.pt",
        ],
        "md5": "3e82b56bfdd4c191a8c61bbed59fcf16",
    },
    {
        "run": "e4p__mosaic-close50__s0",
        "family": "postsppf",
        "epochs": 100,
        "close_mosaic": 50,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e4p__mosaic-close50__s0/weights/best.pt",
            "runs/e4p__mosaic-close50__s0/weights/best.pt",
        ],
        "md5": "12a8df80659896a248e2b8b052983f91",
    },
    # Regularization Lambda Grid (E6')
    {
        "run": "e6p__lam-1e-5__s0",
        "family": "lam_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e6p__lam-1e-5__s0/weights/best.pt", "runs/e6p__lam-1e-5__s0/weights/best.pt"],
        "md5": "a8fc3ba8d37cc603e7ede72e47999285",
    },
    {
        "run": "e6p__lam-1e-4__s0",
        "family": "lam_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e6p__lam-1e-4__s0/weights/best.pt", "runs/e6p__lam-1e-4__s0/weights/best.pt"],
        "md5": "ca93e6ec41d58f9e2056a2d90ba2c137",
    },
    {
        "run": "e6p__lam-1e-4__s1",
        "family": "lam_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 1,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e6p__lam-1e-4__s1/weights/best.pt", "runs/e6p__lam-1e-4__s1/weights/best.pt"],
        "md5": "13851f72f1949fc3b38bf5f1a697b221",
    },
    {
        "run": "e6p__lam-1e-4__s2",
        "family": "lam_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 2,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e6p__lam-1e-4__s2/weights/best.pt", "runs/e6p__lam-1e-4__s2/weights/best.pt"],
        "md5": "f42c349b62962da9602b46eefb311890",
    },
    {
        "run": "e6p__lam-1e-3__s0",
        "family": "lam_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e6p__lam-1e-3__s0/weights/best.pt", "runs/e6p__lam-1e-3__s0/weights/best.pt"],
        "md5": "c3400f31e425e946c8adcc39894c73ec",
    },
    {
        "run": "e6p__lam-1e-2__s0",
        "family": "lam_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e6p__lam-1e-2__s0/weights/best.pt", "runs/e6p__lam-1e-2__s0/weights/best.pt"],
        "md5": "85a676bfefaebd48ab03865f449b681c",
    },
    {
        "run": "e6p__lam-1e-1__s0",
        "family": "lam_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e6p__lam-1e-1__s0/weights/best.pt", "runs/e6p__lam-1e-1__s0/weights/best.pt"],
        "md5": "70cc24a6dab65c6d858b0653ba1239b8",
    },
    # NWD Formulations (E7')
    {
        "run": "e7p__nwd-calib__s0",
        "family": "nwd_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e7p__nwd-calib__s0/weights/best.pt",
            "runs/e7p__nwd-calib__s0/weights/best.pt",
        ],
        "md5": "15c0c92234f64e3d8648d79e71e2c774",
    },
    {
        "run": "e7p__nwd-scaleinv__s0",
        "family": "nwd_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e7p__nwd-scaleinv__s0/weights/best.pt",
            "runs/e7p__nwd-scaleinv__s0/weights/best.pt",
        ],
        "md5": "afbc9edda9cf145d205a34b27795347d",
    },
    {
        "run": "e7p__nwd-calib-norsl__s0",
        "family": "nwd_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e7p__nwd-calib-norsl__s0/weights/best.pt",
            "runs/e7p__nwd-calib-norsl__s0/weights/best.pt",
        ],
        "md5": "e20dc2b3680a5aed9e25965f226b67dc",
    },
    {
        "run": "runs_fixed/e7p__nwd-sizegate__s0",
        "family": "nwd_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs_fixed/e7p__nwd-sizegate__s0/weights/best.pt",
            "runs_fixed/e7p__nwd-sizegate__s0/weights/best.pt",
            "runs_fixed/detect/runs/e7p__nwd-sizegate__s0/weights/best.pt",
        ],
        "md5": "bd1d7c205b8ccc34c458f2c71471bfaf",
    },
    {
        "run": "runs/detect/runs/e7p__nwd-sizegate__s0",
        "family": "nwd_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": [
            "runs/detect/runs/e7p__nwd-sizegate__s0/weights/best.pt",
            "runs/e7p__nwd-sizegate__s0/weights/best.pt",
        ],
        "md5": "cdb350047dcd37b62f4314afa62f601d",
    },
    # Depth / Integration Position (E3')
    {
        "run": "e3p__pos-p3__s0",
        "family": "pos_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-p3.yaml",
        "candidates": ["runs/detect/runs/e3p__pos-p3__s0/weights/best.pt", "runs/e3p__pos-p3__s0/weights/best.pt"],
        "md5": "002731896f44a04eb892a7b884c5b6bb",
    },
    {
        "run": "e3p__pos-p4__s0",
        "family": "pos_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-p4.yaml",
        "candidates": ["runs/detect/runs/e3p__pos-p4__s0/weights/best.pt", "runs/e3p__pos-p4__s0/weights/best.pt"],
        "md5": "51055040f8e611625dbc82f67e0b6c12",
    },
    {
        "run": "e3p__pos-presppf__s0",
        "family": "pos_sweep",
        "epochs": 100,
        "close_mosaic": 10,
        "seed": 0,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-presppf.yaml",
        "candidates": [
            "runs/detect/runs/e3p__pos-presppf__s0/weights/best.pt",
            "runs/e3p__pos-presppf__s0/weights/best.pt",
        ],
        "md5": "b8149c23f6e6752baa12b3648d869408",
    },
]

# ---------------------------------------------------------------------------
# New Training Catalog (R1.2): 8 Mandatory + 4 Priority 2 (Mosaic)
# ---------------------------------------------------------------------------
MANDATORY_NEW_RUNS: List[Dict] = [
    # 40 epochs (auto -> AdamW lr0=0.000625, momentum=0.9, warmup_bias_lr=0.0)
    {
        "run": "e5p__baseline-ep40__s1",
        "family": "baseline",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 1,
        "optimizer": "auto",
        "cfg": "yolov8s.yaml",
        "candidates": ["runs/detect/runs/e5p__baseline-ep40__s1/weights/best.pt"],
    },
    {
        "run": "e5p__baseline-ep40__s2",
        "family": "baseline",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 2,
        "optimizer": "auto",
        "cfg": "yolov8s.yaml",
        "candidates": ["runs/detect/runs/e5p__baseline-ep40__s2/weights/best.pt"],
    },
    {
        "run": "e5p__postsppf-ep40__s1",
        "family": "postsppf",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 1,
        "optimizer": "auto",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e5p__postsppf-ep40__s1/weights/best.pt"],
    },
    {
        "run": "e5p__postsppf-ep40__s2",
        "family": "postsppf",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 2,
        "optimizer": "auto",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e5p__postsppf-ep40__s2/weights/best.pt"],
    },
    # 60 epochs (auto -> AdamW lr0=0.000625, momentum=0.9, warmup_bias_lr=0.0)
    {
        "run": "e5p__baseline-ep60__s1",
        "family": "baseline",
        "epochs": 60,
        "close_mosaic": 10,
        "seed": 1,
        "optimizer": "auto",
        "cfg": "yolov8s.yaml",
        "candidates": ["runs/detect/runs/e5p__baseline-ep60__s1/weights/best.pt"],
    },
    {
        "run": "e5p__baseline-ep60__s2",
        "family": "baseline",
        "epochs": 60,
        "close_mosaic": 10,
        "seed": 2,
        "optimizer": "auto",
        "cfg": "yolov8s.yaml",
        "candidates": ["runs/detect/runs/e5p__baseline-ep60__s2/weights/best.pt"],
    },
    {
        "run": "e5p__postsppf-ep60__s1",
        "family": "postsppf",
        "epochs": 60,
        "close_mosaic": 10,
        "seed": 1,
        "optimizer": "auto",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e5p__postsppf-ep60__s1/weights/best.pt"],
    },
    {
        "run": "e5p__postsppf-ep60__s2",
        "family": "postsppf",
        "epochs": 60,
        "close_mosaic": 10,
        "seed": 2,
        "optimizer": "auto",
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e5p__postsppf-ep60__s2/weights/best.pt"],
    },
]

# Control series ep40-sgd (6 runs: epochs=40, optimizer=SGD, lr0=0.01, seeds 0, 1, 2)
CONTROL_EP40_SGD_RUNS: List[Dict] = [
    {
        "run": "e8p__baseline-ep40sgd__s0",
        "family": "baseline",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 0,
        "optimizer": "SGD",
        "lr0": 0.01,
        "cfg": "yolov8s.yaml",
        "candidates": ["runs/detect/runs/e8p__baseline-ep40sgd__s0/weights/best.pt"],
    },
    {
        "run": "e8p__baseline-ep40sgd__s1",
        "family": "baseline",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 1,
        "optimizer": "SGD",
        "lr0": 0.01,
        "cfg": "yolov8s.yaml",
        "candidates": ["runs/detect/runs/e8p__baseline-ep40sgd__s1/weights/best.pt"],
    },
    {
        "run": "e8p__baseline-ep40sgd__s2",
        "family": "baseline",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 2,
        "optimizer": "SGD",
        "lr0": 0.01,
        "cfg": "yolov8s.yaml",
        "candidates": ["runs/detect/runs/e8p__baseline-ep40sgd__s2/weights/best.pt"],
    },
    {
        "run": "e8p__postsppf-ep40sgd__s0",
        "family": "postsppf",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 0,
        "optimizer": "SGD",
        "lr0": 0.01,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e8p__postsppf-ep40sgd__s0/weights/best.pt"],
    },
    {
        "run": "e8p__postsppf-ep40sgd__s1",
        "family": "postsppf",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 1,
        "optimizer": "SGD",
        "lr0": 0.01,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e8p__postsppf-ep40sgd__s1/weights/best.pt"],
    },
    {
        "run": "e8p__postsppf-ep40sgd__s2",
        "family": "postsppf",
        "epochs": 40,
        "close_mosaic": 10,
        "seed": 2,
        "optimizer": "SGD",
        "lr0": 0.01,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e8p__postsppf-ep40sgd__s2/weights/best.pt"],
    },
]

# Reproducibility check run (e5p__baseline-ep40__s0 repeated as e9p)
REPRO_CHECK_RUN: Dict = {
    "run": "e9p__repro-baseline-ep40__s0",
    "family": "baseline",
    "epochs": 40,
    "close_mosaic": 10,
    "seed": 0,
    "optimizer": "auto",
    "cfg": "yolov8s.yaml",
    "candidates": ["runs/detect/runs/e9p__repro-baseline-ep40__s0/weights/best.pt"],
    "target_mAP50": 0.7159,
    "tolerance": 0.003,  # +-0.3 pp
}

MOSAIC_PRIORITY2_RUNS: List[Dict] = [
    {
        "run": "e4p__mosaic-close0__s1",
        "family": "postsppf",
        "epochs": 100,
        "close_mosaic": 0,
        "seed": 1,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e4p__mosaic-close0__s1/weights/best.pt"],
    },
    {
        "run": "e4p__mosaic-close0__s2",
        "family": "postsppf",
        "epochs": 100,
        "close_mosaic": 0,
        "seed": 2,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e4p__mosaic-close0__s2/weights/best.pt"],
    },
    {
        "run": "e4p__mosaic-close50__s1",
        "family": "postsppf",
        "epochs": 100,
        "close_mosaic": 50,
        "seed": 1,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e4p__mosaic-close50__s1/weights/best.pt"],
    },
    {
        "run": "e4p__mosaic-close50__s2",
        "family": "postsppf",
        "epochs": 100,
        "close_mosaic": 50,
        "seed": 2,
        "cfg": "ultralytics/cfg/models/v8/yolov8s-rfd-postsppf.yaml",
        "candidates": ["runs/detect/runs/e4p__mosaic-close50__s2/weights/best.pt"],
    },
]


def compute_file_md5(path: Path) -> str:
    """Compute MD5 hash for a file efficiently."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_checkpoint(candidates: List[str], run_name: str, root: Path = REPO_ROOT) -> Optional[Path]:
    """Resolve checkpoint file path from candidates or filesystem search."""
    for c in candidates:
        p = root / c if not Path(c).is_absolute() else Path(c)
        if p.is_file():
            return p
    # Search only under runs and runs_fixed if candidates not found
    clean_name = run_name.split("/")[-1]
    matches = []
    for sdir in [root / "runs", root / "runs_fixed"]:
        if sdir.is_dir():
            matches.extend(list(sdir.glob(f"**/{clean_name}/weights/best.pt")))
    if matches:
        if "fixed" in run_name:
            fixed_matches = [m for m in matches if "fixed" in str(m)]
            if fixed_matches:
                return fixed_matches[0]
        else:
            non_fixed = [m for m in matches if "fixed" not in str(m)]
            if non_fixed:
                return non_fixed[0]
        return matches[0]
    return None


def resolve_data_yaml(data_arg: str, root: Path = REPO_ROOT) -> Path:
    """Resolve exdark.yaml or dataset yaml path safely."""
    p = Path(data_arg)
    if not p.is_absolute():
        p = root / p
    if p.is_file():
        return p
    # Fallback to REPO_ROOT / exdark.yaml
    fallback = root / "exdark.yaml"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Dataset config not found at '{data_arg}' or '{fallback}'")
