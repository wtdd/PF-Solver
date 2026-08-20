import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from GNSMsg_armijo import GNSMsg
from GNSMsg_SelfAttention_armijo import GNSMsg_EdgeSelfAttn
from pignn_attn_ls_data import (
    CommonCSVPowerFlowDataset,
    SharedTopologyCollator,
    data_system_name,
    evaluate_pignn_attn_ls_heatmap_cells,
    evaluate_pignn_attn_ls_loader,
    forward_pignn_attn_ls_model,
    move_batch_to_device,
    print_heatmap_cell_summary,
    print_ill_conditioned_summary,
    print_metric_summary,
    print_region_mismatch_summary,
    split_train_val,
    voltage_supervised_loss,
    write_heatmap_cell_csv,
)


class Tee:
    def __init__(self, stream, log_path: Path):
        self.stream = stream
        self.file = log_path.open("w", encoding="utf-8", buffering=1)

    def write(self, text):
        try:
            self.stream.write(text)
        except UnicodeEncodeError:
            encoding = getattr(self.stream, "encoding", None) or "utf-8"
            self.stream.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))
        self.file.write(text)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def start_log(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tee = Tee(sys.stdout, log_path)
    sys.stdout = tee
    sys.stderr = tee
    print(f"Log file           : {log_path.name}")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y", "t"):
        return True
    if value in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

# PIGNN-Attn-LS
def parse_args():
    default_data = Path(__file__).resolve().parents[1] / "Data" / "IEEE_118"
    parser = argparse.ArgumentParser(description="Train/test PIGNN-Attn-LS on Slover common CSV data.")
    parser.add_argument("--data_dir", type=str, default=str(default_data))
    parser.add_argument("--epochs", type=int, default=200, help="Paper training schedule is 40 epochs.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", choices=("cosine", "plateau", "none"), default="cosine")
    parser.add_argument("--cosine_tmax", type=int, default=40)
    parser.add_argument("--scheduler_patience", type=int, default=10)
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--disable_early_stop", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tol", type=float, default=1e-3)

    parser.add_argument("--model", choices=("GNSMsg_EdgeSelfAttn", "GNSMsg"), default="GNSMsg_EdgeSelfAttn")
    parser.add_argument("--d", type=int, default=4)
    parser.add_argument("--d_hi", type=int, default=16)
    parser.add_argument("--K", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_attn_layers", type=int, default=1)
    parser.add_argument("--use_armijo", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--vlimit", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--batch_mode", choices=("plain", "blockdiag"), default="plain",
        help="For one fixed IEEE topology, plain is mathematically equivalent and avoids a huge dense block matrix.",
    )
    parser.add_argument("--no_pinn", action="store_true", help="Ablation: use supervised voltage loss only.")
    parser.add_argument("--supervised_weight", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when supported (optional warm-up cost).")
    parser.add_argument("--rebuild_data_cache", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--load_trained_model", nargs="?", const=True, default=False, type=str2bool,
        help="Load an existing checkpoint and skip training, matching SOTA4's execution mode.",
    )
    return parser.parse_args()


def choose_device(which: str) -> torch.device:
    if which == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(which)


def data_system_tag(data_dir: str) -> str:
    return "".join(ch for ch in data_system_name(data_dir).lower() if ch.isalnum())


def choose_log_path(args, ckpt_path: Path) -> Path:
    if args.eval_only or (args.load_trained_model and ckpt_path.exists()):
        return ckpt_path.parent / f"{data_system_tag(args.data_dir)}_test.log"
    return ckpt_path.with_suffix(ckpt_path.suffix + ".log")


def build_model(args, pinn: bool) -> torch.nn.Module:
    if args.model == "GNSMsg":
        return GNSMsg(d=args.d, d_hi=args.d_hi, K=args.K, pinn=pinn, gamma=args.gamma, v_limit=args.vlimit, use_armijo=args.use_armijo)
    return GNSMsg_EdgeSelfAttn(
        d=args.d, d_hi=args.d_hi, n_heads=args.n_heads, K=args.K, pinn=pinn,
        gamma=args.gamma, v_limit=args.vlimit, use_armijo=args.use_armijo,
        num_attn_layers=args.num_attn_layers,
    )


def make_loader(dataset, batch_size: int, shuffle: bool, args, *, force_plain: bool = False):
    workers = max(int(args.num_workers), 0)
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=SharedTopologyCollator(
            dataset, blockdiag=(args.batch_mode == "blockdiag" and not force_plain)
        ),
        pin_memory=torch.cuda.is_available(),
    )
    if workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


def make_scheduler(optimizer, args):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(args.cosine_tmax), 1), eta_min=args.min_lr,
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.scheduler_patience,
        threshold=args.min_delta, threshold_mode="abs", min_lr=args.min_lr,
    )


def run_epoch(model, loader, optimizer, device, *, train: bool, pinn: bool, supervised_weight: float, grad_clip: float):
    model.train(train)
    totals = {"loss": 0.0, "phys": 0.0, "sup": 0.0}
    total_graphs = 0
    context = torch.enable_grad() if train else torch.inference_mode()
    with context:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            n_graphs = int(batch["sizes"].numel()) if "sizes" in batch else int(batch["S_start"].shape[0])
            out = forward_pignn_attn_ls_model(model, batch, pinn=pinn)
            v_pred, phys_loss = out
            sup_loss = voltage_supervised_loss(v_pred, batch["V_newton"])
            if pinn:
                loss = phys_loss + float(supervised_weight) * sup_loss
            else:
                phys_loss = v_pred.new_zeros(())
                loss = sup_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            total_graphs += n_graphs
            totals["loss"] += float(loss.detach()) * n_graphs
            totals["phys"] += float(phys_loss.detach()) * n_graphs
            totals["sup"] += float(sup_loss.detach()) * n_graphs
    return {key: value / max(total_graphs, 1) for key, value in totals.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    pinn = not args.no_pinn

    script_dir = Path(__file__).resolve().parent
    ckpt_dir = script_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt).expanduser().resolve() if args.ckpt else ckpt_dir / f"{data_system_tag(args.data_dir)}_PIGNN-Attn-LS_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    start_log(choose_log_path(args, ckpt_path))

    train_full = CommonCSVPowerFlowDataset(args.data_dir, "train", rebuild_cache=args.rebuild_data_cache)
    heatmap_test_ds = CommonCSVPowerFlowDataset(args.data_dir, "test", exclude_source="ill-conditioned")
    ill_test_ds = CommonCSVPowerFlowDataset(args.data_dir, "test", source_filter="ill-conditioned", allow_empty=True)
    train_ds, val_ds = split_train_val(train_full, args.val_ratio, args.seed)
    train_loader = make_loader(train_ds, args.batch_size, True, args)
    val_loader = make_loader(val_ds, args.batch_size, False, args)
    # Per-sample IDs/levels are required by SOTA4-compatible reports; use the
    # true batch layout for evaluation even if legacy blockdiag training is selected.
    test_loader = make_loader(heatmap_test_ds, args.test_batch_size, False, args, force_plain=True)
    ill_loader = make_loader(ill_test_ds, args.test_batch_size, False, args, force_plain=True)

    print(f"Using device       : {device}")
    print(f"Data directory     : {data_system_name(args.data_dir)}")
    print(f"Model              : {args.model}")
    print(f"Paper config       : d={args.d}, hidden={args.d_hi}, K={args.K}, heads={args.n_heads}, attn_layers={args.num_attn_layers}")
    print(f"PINN/supervised wt : {pinn}/{args.supervised_weight:g}")
    print(f"Armijo/V-limit     : {args.use_armijo}/{args.vlimit}")
    print(f"Checkpoint         : {ckpt_path.name}")
    print(f"Dataset sizes      : train={len(train_ds)}, val={len(val_ds)}, heatmap_test={len(heatmap_test_ds)}, ill_test={len(ill_test_ds)}")
    print(f"Batch mode         : {args.batch_mode} (shared fixed topology)")
    print(f"Scheduler          : {args.scheduler}")

    model = build_model(args, pinn=pinn).to(device)
    if args.compile:
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile       : enabled")
        else:
            print("torch.compile       : unavailable; continuing eagerly")

    should_load = bool((args.eval_only or args.load_trained_model) and ckpt_path.exists())
    if args.eval_only and not ckpt_path.exists():
        raise FileNotFoundError(f"Evaluation checkpoint not found: {ckpt_path}")
    if args.load_trained_model and not ckpt_path.exists():
        print("Requested checkpoint does not exist; starting training.")
    if not should_load:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = make_scheduler(optimizer, args)
        early_patience = args.early_stop_patience or max(args.scheduler_patience * 2, 1)
        best_val, bad_validations = math.inf, 0
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_stats = run_epoch(model, train_loader, optimizer, device, train=True, pinn=pinn, supervised_weight=args.supervised_weight, grad_clip=args.grad_clip)
            do_val = epoch == 1 or epoch % max(args.val_interval, 1) == 0 or epoch == args.epochs
            val_stats = run_epoch(model, val_loader, optimizer, device, train=False, pinn=pinn, supervised_weight=args.supervised_weight, grad_clip=args.grad_clip) if do_val else None
            previous_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None:
                if args.scheduler == "plateau":
                    if do_val:
                        scheduler.step(val_stats["loss"])
                else:
                    scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            val_text = (
                f"val loss={val_stats['loss']:.6e} phys={val_stats['phys']:.6e} sup={val_stats['sup']:.6e}"
                if val_stats else "val skipped"
            )
            print(
                f"Epoch {epoch:04d}/{args.epochs} | train loss={train_stats['loss']:.6e} "
                f"phys={train_stats['phys']:.6e} sup={train_stats['sup']:.6e} | {val_text} | "
                f"lr={current_lr:.3e} | time={time.perf_counter() - start:.2f}s"
            )
            if current_lr < previous_lr:
                print(f"🔻 LR reduced         : {previous_lr:.3e} -> {current_lr:.3e}")
            if do_val and val_stats["loss"] < best_val - args.min_delta:
                best_val, bad_validations = val_stats["loss"], 0
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                torch.save({"model_state_dict": raw_model.state_dict(), "args": vars(args), "best_val_loss": best_val}, ckpt_path)
                print(f"🟩 Saved best checkpoint: {ckpt_path.name}")
            elif do_val:
                bad_validations += 1
                print(f"🔘 No improvement     : {bad_validations}/{early_patience} (best={best_val:.6e})")
                if not args.disable_early_stop and bad_validations >= early_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
    else:
        print("Load checkpoint mode: skip training.")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found after training/load decision: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(state_dict)
    print("Checkpoint loaded  : OK")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    test_start = time.perf_counter()
    metrics = evaluate_pignn_attn_ls_loader(model, test_loader, device, tol=args.tol, pinn=pinn)
    cells = evaluate_pignn_attn_ls_heatmap_cells(model, test_loader, device, model_steps=float(args.K), pinn=pinn)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print_metric_summary(f"PIGNN-Attn-LS test metrics (tol={args.tol:g})", metrics)
    print(f"Test evaluation time             : {time.perf_counter() - test_start:.6f} s")
    print_heatmap_cell_summary(cells)
    print_region_mismatch_summary(cells)
    heatmap_csv = ckpt_path.parent / f"{data_system_tag(args.data_dir)}.csv"
    write_heatmap_cell_csv(cells, heatmap_csv)

    print("Testing ill-conditioned: running success metrics", flush=True)
    ill_csv = ckpt_path.parent / f"{data_system_tag(args.data_dir).upper()}_ill.csv"
    ill_metrics = (
        evaluate_pignn_attn_ls_loader(
            model, ill_loader, device, tol=args.tol, pinn=pinn,
            nr_refine_max_steps=20, nr_refine_tol=1e-6, mismatch_csv_path=ill_csv,
        ) if len(ill_test_ds) else None
    )
    print_ill_conditioned_summary(ill_metrics)


if __name__ == "__main__":
    main()
