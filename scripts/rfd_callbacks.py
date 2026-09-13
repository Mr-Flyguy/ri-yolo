# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ultralytics training callbacks for RFD module monitoring and logging.

Records gamma gating weights and illumination map statistics at the end of each epoch.
"""

import csv
import os


def add_rfd_logging(trainer):
    """Callback triggered on fit epoch end to log RFD parameters and illumination stats."""
    save_dir = getattr(trainer, "save_dir", "runs")
    path = os.path.join(save_dir, "rfd_log.csv")
    is_new = not os.path.exists(path)
    rows = []

    model_obj = getattr(trainer, "model", None)
    if model_obj is None:
        return

    for name, m in model_obj.named_modules():
        if m.__class__.__name__ not in ("RFDBlock", "RFDBlockNoGate"):
            continue

        gamma_val = float(m.gamma.detach().cpu().reshape(-1)[0]) if hasattr(m, "gamma") else 0.0
        r = {
            "epoch": getattr(trainer, "epoch", 0),
            "module": name,
            "gamma": gamma_val,
        }

        # Check for detached stat copy or active illumination map
        ill = getattr(m, "_ill_stat", None)
        if ill is None:
            ill = getattr(m, "_illumination_map", None)

        if ill is not None:
            ill_f = ill.detach().float()
            r.update({
                "L_mean": float(ill_f.mean()),
                "L_std_spatial": float(ill_f.std(dim=(2, 3)).mean()) if ill_f.ndim >= 4 else float(ill_f.std()),
                "L_min": float(ill_f.min()),
                "L_max": float(ill_f.max()),
                "L_sat_frac": float(((ill_f < 0.02) | (ill_f > 0.98)).float().mean()),
            })
        rows.append(r)

    if not rows:
        return

    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if is_new:
            w.writeheader()
        w.writerows(rows)
