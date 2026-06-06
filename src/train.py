"""Train the CenterNet-style GUV detector on the synthetic dataset.

    uv run python src/train.py --config configs/train.yaml
    uv run python src/train.py --config configs/train.yaml --smoke   # tiny, fast

Loss = penalty-reduced focal loss on the center heatmap + L1 on the radius at GT
centers. AdamW; LR = linear warmup then cosine decay to ~0. Mixed precision
(autocast + GradScaler) for the H100. Logs train/val loss each epoch, saves the
best checkpoint by val loss, and dumps a few decoded val predictions each epoch
so you can watch it learn.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import GUVDataset  # noqa: E402
from src.detect import decode  # noqa: E402
from src.model import GUVNet  # noqa: E402


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
_FOCAL_EPS = 1e-4


def focal_loss(pred: torch.Tensor, gt: torch.Tensor, alpha: float = 2.0,
               beta: float = 4.0, eps: float = _FOCAL_EPS) -> torch.Tensor:
    """CenterNet penalty-reduced focal loss on a [0,1] heatmap.

    Positives are exact GT centers (gt == 1); everything else is negative, with
    the penalty down-weighted near a center by (1 - gt)**beta. Normalized by the
    number of positives.

    Numerical stability (this is the classic focal-loss-from-scratch NaN):
      - compute in float32 (fp16 log/exp/pow under AMP is the usual NaN source);
      - clamp the predicted probabilities to [eps, 1 - eps] so log() never sees
        exactly 0 or 1.
    """
    pred = pred.float().clamp(eps, 1.0 - eps)  # fp32 + clamp before any log()
    gt = gt.float()

    pos = gt.eq(1).float()
    neg = 1.0 - pos
    neg_weights = torch.pow(1.0 - gt, beta)

    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg

    n_pos = pos.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if n_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / n_pos


def radius_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 on radius, supervised ONLY where mask == 1 (true centers). fp32."""
    pred = pred.float()
    target = target.float()
    mask = mask.float()
    loss = (torch.abs(pred - target) * mask).sum()
    return loss / (mask.sum() + 1e-4)


# --------------------------------------------------------------------------- #
# LR schedule: linear warmup -> cosine decay to ~0
# --------------------------------------------------------------------------- #
def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def save_val_viz(model, dataset, cfg, device, out_path, n: int = 4):
    """Decode predictions on a few val images; save heatmap + circles overlay."""
    model.eval()
    n = min(n, len(dataset))
    if n == 0:
        return
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    axes = np.atleast_2d(axes)
    with torch.no_grad():
        for k in range(n):
            sample = dataset[k]
            img = sample["image"].to(device)[None]  # (1,1,H,W)
            out = model(img)
            hm = out["hm"][0, 0].cpu().numpy()
            dets = decode(out["hm"][0], out["radius"][0],
                          threshold=cfg["detect"]["threshold"],
                          nms_dist=cfg["detect"]["nms_dist"],
                          down_ratio=cfg["model"]["out_stride"])
            img_np = sample["image"][0].cpu().numpy()
            axes[0, k].imshow(img_np, cmap="gray")
            for d in dets:
                axes[0, k].add_patch(Circle((d[0], d[1]), d[2], fill=False, edgecolor="red", lw=1))
            axes[0, k].set_title(f"pred: {len(dets)} GUVs")
            axes[0, k].axis("off")
            axes[1, k].imshow(hm, cmap="magma", vmin=0, vmax=1)
            axes[1, k].set_title("heatmap")
            axes[1, k].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Train / eval epochs
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, cfg, device, optimizer=None, scheduler=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    hm_w = cfg["loss"]["heatmap_weight"]
    r_w = cfg["loss"]["radius_weight"]
    use_amp = cfg["train"]["amp"] and device.startswith("cuda")
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    totals = {"loss": 0.0, "hm": 0.0, "radius": 0.0}
    n = 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        hm_gt = batch["heatmap"].to(device, non_blocking=True)
        r_gt = batch["radius_map"].to(device, non_blocking=True)
        mask = batch["reg_mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            # Forward (and backward) use AMP; the LOSS math is forced to float32.
            with torch.autocast(device_type, enabled=use_amp):
                out = model(img)
            with torch.autocast(device_type, enabled=False):
                hm_pred = out["hm"].float()
                r_pred = out["radius"].float()
                l_hm = focal_loss(hm_pred, hm_gt.float(),
                                  cfg["loss"]["focal_alpha"], cfg["loss"]["focal_beta"])
                l_r = radius_l1(r_pred, r_gt.float(), mask.float())
                loss = hm_w * l_hm + r_w * l_r

        if train:
            # Guard: never let a single non-finite batch poison the weights.
            if not torch.isfinite(loss):
                print(f"WARNING: non-finite loss (hm={l_hm.item()}, radius={l_r.item()}); "
                      f"skipping optimizer step for this batch")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # unscale so clipping sees real grads
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # If inf grads were found, scaler skips the step and lowers the
                # scale -> don't advance the scheduler (avoids the "scheduler
                # before optimizer" warning).
                stepped = scaler.get_scale() >= prev_scale
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                stepped = True
            if scheduler is not None and stepped:
                scheduler.step()  # AFTER a real optimizer.step()

        bs = img.size(0)
        totals["loss"] += loss.item() * bs
        totals["hm"] += l_hm.item() * bs
        totals["radius"] += l_r.item() * bs
        n += bs

    return {k: v / max(1, n) for k, v in totals.items()}


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "configs" / "train.yaml"))
    parser.add_argument("--smoke", action="store_true", help="tiny subset, few epochs, to test the loop")
    parser.add_argument("--device", default=None, help="override train.device (e.g. cuda, cpu)")
    parser.add_argument("--num-workers", type=int, default=None, help="override data.num_workers")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device is not None:
        cfg["train"]["device"] = args.device
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers

    if args.smoke:
        cfg["train"]["epochs"] = min(cfg["train"].get("smoke_epochs", 2), cfg["train"]["epochs"])
        cfg["data"]["limit"] = cfg["train"].get("smoke_limit", 50)
        cfg["train"]["batch_size"] = min(cfg["train"]["batch_size"], 8)
        cfg["data"]["num_workers"] = 0

    device = cfg["train"]["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        device = "cpu"

    root = repo_root / cfg["data"]["root"]
    limit = cfg["data"].get("limit")
    ds_kw = dict(
        down_ratio=cfg["model"]["out_stride"],
        sigma_scale=cfg["data"]["sigma_scale"],
        sigma_min=cfg["data"]["sigma_min"],
        sigma_max=cfg["data"]["sigma_max"],
    )
    train_ds = GUVDataset(root, "train", limit=limit, **ds_kw)
    val_ds = GUVDataset(root, "val", limit=(limit if limit is None else max(8, limit // 5)), **ds_kw)
    print(f"train={len(train_ds)} val={len(val_ds)} device={device}")

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                              num_workers=cfg["data"]["num_workers"], pin_memory=device.startswith("cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                            num_workers=cfg["data"]["num_workers"], pin_memory=device.startswith("cuda"))

    model = GUVNet(in_ch=1, base=cfg["model"]["base"], depth=cfg["model"]["depth"],
                   out_stride=cfg["model"]["out_stride"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg["train"]["epochs"]
    warmup_steps = int(cfg["train"]["warmup_frac"] * total_steps) if cfg["train"].get("warmup_frac") \
        else cfg["train"].get("warmup_steps", steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, make_lr_lambda(warmup_steps, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.startswith("cuda"))

    out_dir = repo_root / cfg["train"]["out_dir"]
    (out_dir / "viz").mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    model_cfg = {"base": cfg["model"]["base"], "depth": cfg["model"]["depth"], "out_stride": cfg["model"]["out_stride"]}
    for epoch in range(cfg["train"]["epochs"]):
        tr = run_epoch(model, train_loader, cfg, device, optimizer, scheduler, scaler)
        va = run_epoch(model, val_loader, cfg, device)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch+1}/{cfg['train']['epochs']} "
              f"lr={lr_now:.2e} | train loss={tr['loss']:.4f} (hm={tr['hm']:.4f} r={tr['radius']:.4f}) "
              f"| val loss={va['loss']:.4f} (hm={va['hm']:.4f} r={va['radius']:.4f})")

        save_val_viz(model, val_ds, cfg, device, out_dir / "viz" / f"epoch_{epoch+1:03d}.png",
                     n=cfg["train"].get("viz_n", 4))

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save({"model_state": model.state_dict(), "model_cfg": model_cfg,
                        "epoch": epoch + 1, "val_loss": best_val},
                       out_dir / "best.pt")
            print(f"  saved best (val loss={best_val:.4f})")

    torch.save({"model_state": model.state_dict(), "model_cfg": model_cfg,
                "epoch": cfg["train"]["epochs"], "val_loss": best_val}, out_dir / "last.pt")
    print(f"done. best val loss={best_val:.4f}. checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
