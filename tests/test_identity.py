# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Identity mapping test at initialization (gamma=0).

Validates that inserting RFD/RFDBlockNoGate blocks does not alter the pretrained model's
predictions when gamma parameter is zero, confirming correct weight transfer and layer indexing.
"""

import argparse
import sys
import torch
from ultralytics import YOLO

CFGS = [
    "yolov8s-rfd-p3.yaml",
    "yolov8s-rfd-p4.yaml",
    "yolov8s-rfd-presppf.yaml",
    "yolov8s-rfd-postsppf.yaml",
    "yolov8s-capctrl.yaml",
]

try:
    import pytest
    _parametrize = pytest.mark.parametrize("cfg", CFGS)
except ImportError:
    def _parametrize(fn):
        return fn


def check_identity_at_init(cfg, weights="yolov8s.pt", atol=1e-4):
    """Compute and verify identity output difference for RFD module at initialization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = YOLO(weights).model.to(device).eval()
    mod = YOLO(cfg).load(weights).model.to(device).eval()

    torch.manual_seed(42)
    x = torch.randn(1, 3, 640, 640, device=device)

    with torch.no_grad():
        a = base(x)[0]
        b = mod(x)[0]

    assert a.shape == b.shape, f"{cfg}: shape mismatch {a.shape} vs {b.shape}"
    max_diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=atol), f"{cfg}: identity broken, max|d|={max_diff:.2e} (atol={atol})"
    return max_diff


@_parametrize
def test_identity_at_init(cfg, weights="yolov8s.pt", atol=1e-4):
    """Test that model output with RFD module matches baseline output at initialization."""
    check_identity_at_init(cfg, weights=weights, atol=atol)


def main():
    parser = argparse.ArgumentParser(description="Test identity mapping of RFD models at initialization.")
    parser.add_argument("--weights", default="yolov8s.pt", help="Path to baseline pretrained weights")
    parser.add_argument("--cfgs", nargs="+", default=CFGS, help="List of model config files to test")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance threshold")
    args = parser.parse_args()

    print(f"Testing identity at init with baseline weights '{args.weights}' (atol={args.atol}):")
    all_passed = True
    for cfg in args.cfgs:
        try:
            max_d = check_identity_at_init(cfg, weights=args.weights, atol=args.atol)
            print(f"  [PASS] {cfg:30s} max|diff| = {max_d:.2e}")
        except Exception as e:
            print(f"  [FAIL] {cfg:30s} ERROR: {e}")
            all_passed = False

    if not all_passed:
        print("\nERROR: One or more configurations failed identity test!")
        sys.exit(1)
    print("\nSUCCESS: All configurations passed identity check!")


if __name__ == "__main__":
    main()
