import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import (
    CommonPFDataset,
    collate_samples,
    load_system_info,
    load_ybus,
    move_batch,
    resolve_data_dir,
    split_train_val,
)
from metrics import (
    TGN_loss,
    evaluate_full_summary,
    evaluate_heatmap_cells,
    print_heatmap_cell_summary,
    print_ill_conditioned_summary,
    print_metric_summary,
    print_region_mismatch_summary,
    write_heatmap_cell_csv,
)
from model import TypedGraphPowerFlow


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_VERSION = 3
TRAINING_PROTOCOL = "paper_lr_clip_recovery_plateau_v3"


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
    print(f"Log file           : {log_path}")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in ("true", "1", "yes", "y", "t"):
        return True
    if value in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def data_system_tag(data_dir: str) -> str:
    path = Path(data_dir).expanduser().resolve()
    if path.name.lower() == "common":
        path = path.parent
    return "".join(ch for ch in path.name.lower() if ch.isalnum())


def data_system_name(data_dir: str) -> str:
    path = Path(data_dir).expanduser().resolve()
    if path.name.lower() == "common":
        path = path.parent
    return path.name


def choose_log_path(args, ckpt_path: Path, load_compatible_checkpoint: bool) -> Path:
    if args.load_trained_model and load_compatible_checkpoint:
        return ckpt_path.parent / f"{data_system_tag(args.data_dir)}_test.log"
    return ckpt_path.with_suffix(ckpt_path.suffix + ".log")


def load_checkpoint_file(path: Path, map_location):
    """Load trusted local experiment state without relying on PyTorch's changing default."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.6
        return torch.load(path, map_location=map_location)


def expected_model_config(args, info) -> dict:
    return {
        "T": int(args.tgn_layers),
        "L": int(args.message_steps),
        "hidden_dim": int(args.hidden_dim),
        "decoder_init": str(args.decoder_init),
        "n_bus": int(info.n_bus),
        "n_branch": int(info.n_branch),
    }


def inspect_checkpoint(path: Path, args, info):
    """Return (payload, compatible, explanation) for evaluation/skip decisions."""
    if not path.exists():
        return None, False, "checkpoint does not exist"
    try:
        payload = load_checkpoint_file(path, map_location="cpu")
    except Exception as exc:
        return None, False, f"checkpoint cannot be read ({type(exc).__name__}: {exc})"
    if not isinstance(payload, dict) or "model" not in payload:
        return payload, False, "checkpoint has no model state"

    config = payload.get("model_config", {})
    expected = expected_model_config(args, info)
    if payload.get("checkpoint_version") == CHECKPOINT_VERSION and payload.get("training_protocol") == TRAINING_PROTOCOL:
        differences = [
            f"{key}: saved={config.get(key)!r}, requested={value!r}"
            for key, value in expected.items()
            if config.get(key) != value
        ]
        if differences:
            return payload, False, "model configuration differs (" + "; ".join(differences) + ")"
        return payload, True, f"checkpoint v{CHECKPOINT_VERSION}, protocol={TRAINING_PROTOCOL}"

    if args.allow_legacy_checkpoint:
        legacy_expected = {key: expected[key] for key in ("T", "L", "hidden_dim")}
        differences = [
            f"{key}: saved={config.get(key)!r}, requested={value!r}"
            for key, value in legacy_expected.items()
            if config.get(key) != value
        ]
        if differences:
            return payload, False, "legacy model configuration differs (" + "; ".join(differences) + ")"
        return payload, True, "legacy checkpoint explicitly allowed; training protocol cannot be verified"

    saved_version = payload.get("checkpoint_version", "legacy")
    saved_protocol = payload.get("training_protocol", "missing")
    return (
        payload,
        False,
        f"stale training protocol (version={saved_version!r}, protocol={saved_protocol!r}); "
        "use --allow_legacy_checkpoint only for explicit legacy evaluation",
    )


def atomic_save_checkpoint(payload: dict, path: Path) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def make_plateau_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.disable_scheduler:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        # ReduceLROnPlateau reduces when num_bad_epochs > patience. Subtract
        # one so --scheduler_patience=20 means exactly 20 bad epochs.
        patience=args.scheduler_patience - 1,
        threshold=args.min_delta,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )


def effective_early_stop_patience(args) -> int:
    return args.early_stop_patience if args.early_stop_patience is not None else args.scheduler_patience * 2


def run_epoch(model, loader, optimizer, device, info, ybus, train: bool, grad_clip_norm: float):
    model.train(train)
    total = 0.0
    count = 0
    batches = 0
    max_grad_norm = 0.0
    for batch in loader:
        batch = move_batch(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            loss = TGN_loss(model, batch, info, ybus)
            if not bool(torch.isfinite(loss).item()):
                optimizer.zero_grad(set_to_none=True)
                return {
                    "loss": math.inf,
                    "max_grad_norm": max_grad_norm,
                    "valid": False,
                    "batches": batches,
                    "failure": "non-finite loss",
                }
            if train:
                loss.backward()
                clip_limit = grad_clip_norm if grad_clip_norm > 0.0 else math.inf
                raw_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
                if not bool(torch.isfinite(raw_grad_norm).item()):
                    optimizer.zero_grad(set_to_none=True)
                    return {
                        "loss": math.inf,
                        "max_grad_norm": math.inf,
                        "valid": False,
                        "batches": batches,
                        "failure": "non-finite gradient norm",
                    }
                max_grad_norm = max(max_grad_norm, float(raw_grad_norm.item()))
                optimizer.step()
        batch_size = int(batch["p_spec"].shape[0])
        total += float(loss.detach().item()) * batch_size
        count += batch_size
        batches += 1
    if count == 0:
        return {
            "loss": math.inf,
            "max_grad_norm": max_grad_norm,
            "valid": False,
            "batches": 0,
            "failure": "empty data loader",
        }
    return {
        "loss": total / count,
        "max_grad_norm": max_grad_norm,
        "valid": True,
        "batches": batches,
        "failure": None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Physics-informed Typed Graph Network AC power-flow solver")
    parser.add_argument("--data_dir", default=str(ROOT / "Data" / "IEEE_118"))
    # Data split, batching and LR schedule stay aligned with GIN. The initial
    # learning rate follows the TGN paper (Sec. 4.2).
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler_patience", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--min_delta", type=float, default=1e-5, help="Minimum absolute validation-loss decrease counted as an improvement.")
    parser.add_argument("--disable_scheduler", action="store_true")
    parser.add_argument("--disable_early_stop", action="store_true")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--explosion_factor", type=float, default=100.0)
    parser.add_argument("--max_recoveries", type=int, default=5)
    parser.add_argument("--disable_recovery", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-3)
    # Paper model settings (Sec. 4.2): T=15, L=2, d=16.
    parser.add_argument("--tgn_layers", type=int, default=15, help="T: independently parameterized TGN layers")
    parser.add_argument("--message_steps", type=int, default=2, help="L: shared message/update steps per TGN layer")
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument(
        "--decoder_init",
        choices=("tiny", "glorot"),
        default="tiny",
        help="tiny is the stable residual-solver initialization; glorot enables a TensorFlow-style decoder ablation.",
    )
    parser.add_argument("--ckpt", default="")
    parser.add_argument(
        "--load_trained_model",
        nargs="?",
        const=True,
        default=True,
        type=str2bool,
        help="Skip training only when a checkpoint from the current compatible protocol exists.",
    )
    parser.add_argument(
        "--allow_legacy_checkpoint",
        action="store_true",
        help="Explicitly allow evaluation of an unversioned legacy checkpoint; never enabled by default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.lr <= 0.0 or args.min_lr <= 0.0:
        raise ValueError("--lr and --min_lr must be positive")
    if args.scheduler_patience < 1:
        raise ValueError("--scheduler_patience must be at least 1")
    if args.early_stop_patience is not None and args.early_stop_patience < args.scheduler_patience:
        raise ValueError("--early_stop_patience must be at least --scheduler_patience")
    if args.min_delta < 0.0:
        raise ValueError("--min_delta cannot be negative")
    if not 0.0 < args.lr_factor < 1.0:
        raise ValueError("--lr_factor must be between 0 and 1")
    if args.explosion_factor <= 1.0:
        raise ValueError("--explosion_factor must be greater than 1")
    if args.max_recoveries < 0:
        raise ValueError("--max_recoveries cannot be negative")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info = load_system_info(args.data_dir)
    ybus = torch.tensor(load_ybus(args.data_dir), dtype=torch.complex64, device=device)
    train_all = CommonPFDataset(args.data_dir, "train")
    test_set = CommonPFDataset(args.data_dir, "test", exclude_source="ill-conditioned")
    ill_test_set = CommonPFDataset(args.data_dir, "test", source_filter="ill-conditioned", allow_empty=True)
    train_set, val_set = split_train_val(train_all, args.val_ratio, args.seed)
    loader_options = {"collate_fn": collate_samples, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **loader_options)
    ill_test_loader = DataLoader(ill_test_set, batch_size=args.test_batch_size, shuffle=False, **loader_options)

    model = TypedGraphPowerFlow(
        info,
        tgn_layers=args.tgn_layers,
        message_steps=args.message_steps,
        hidden_dim=args.hidden_dim,
        decoder_init=args.decoder_init,
    ).to(device)
    ckpt_dir = Path(__file__).resolve().parent / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_dir / f"{data_system_tag(args.data_dir)}_TGN_best.pt"
    ckpt_path = ckpt_path.expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    _existing_checkpoint, checkpoint_compatible, checkpoint_reason = inspect_checkpoint(ckpt_path, args, info)
    should_load_existing = bool(args.load_trained_model and checkpoint_compatible)
    start_log(choose_log_path(args, ckpt_path, should_load_existing))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_plateau_scheduler(optimizer, args)
    early_patience = effective_early_stop_patience(args)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(f"Using device       : {device}")
    print(f"Data directory     : {resolve_data_dir(args.data_dir)}")
    print("Model              : Typed Graph Network power-flow solver")
    print(f"TGN configuration  : T={args.tgn_layers}, L={args.message_steps}, d={args.hidden_dim}")
    print(f"Decoder init       : {args.decoder_init}")
    print(f"Trainable params   : {parameter_count}")
    print(f"Graph              : buses={info.n_bus}, branches={info.n_branch}, PV={len(info.pv_idx)}, PQ={len(info.pq_idx)}")
    print(f"Checkpoint         : {ckpt_path}")
    print(f"Dataset sizes      : train={len(train_set)}, val={len(val_set)}, heatmap_test={len(test_set)}, ill_test={len(ill_test_set)}")
    print(f"Batch sizes        : train={args.batch_size}, test={args.test_batch_size}")
    print(f"Optimizer          : Adam(lr={args.lr:.3e}, weight_decay={args.weight_decay:.3e})")
    if scheduler is None:
        print("LR schedule        : disabled")
    else:
        print(
            f"LR schedule        : reduce after {args.scheduler_patience} bad epochs "
            f"(factor={args.lr_factor}, min_lr={args.min_lr:.3e})"
        )
    print(f"Gradient clipping  : global norm={args.grad_clip_norm:.3e} (<=0 disables clipping)")
    print(
        f"Explosion recovery : factor>{args.explosion_factor:g}, max_recoveries={args.max_recoveries}, "
        f"enabled={not args.disable_recovery}"
    )
    print(f"Training protocol  : v{CHECKPOINT_VERSION}/{TRAINING_PROTOCOL}")
    print("Training objective : unsupervised AC power-balance MSE (paper Eq. 21)")
    print(f"Load trained model : {args.load_trained_model}")
    print(f"Checkpoint status  : {checkpoint_reason}")

    if should_load_existing:
        print("Load checkpoint mode: compatible checkpoint found; skip training.")
    elif args.load_trained_model:
        print(f"A compatible checkpoint was not found; start training: {ckpt_path}")

    best_val = math.inf
    bad_epochs = 0
    lr_reduced_on_plateau = False
    recoveries = 0
    checkpoint_saved_this_run = False
    initial_model_state = copy.deepcopy(model.state_dict())
    initial_optimizer_state = copy.deepcopy(optimizer.state_dict())
    epochs_to_run = 0 if should_load_existing else args.epochs

    for epoch in range(1, epochs_to_run + 1):
        start = time.time()
        train_stats = run_epoch(
            model, train_loader, optimizer, device, info, ybus, train=True, grad_clip_norm=args.grad_clip_norm
        )
        if train_stats["valid"]:
            val_stats = run_epoch(
                model, val_loader, optimizer, device, info, ybus, train=False, grad_clip_norm=args.grad_clip_norm
            )
        else:
            val_stats = {
                "loss": math.inf,
                "max_grad_norm": 0.0,
                "valid": False,
                "batches": 0,
                "failure": "validation skipped because training failed",
            }
        train_loss = float(train_stats["loss"])
        val_loss = float(val_stats["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"Epoch {epoch:04d}/{epochs_to_run} | lr={current_lr:.3e} | train loss={train_loss:.6e} | "
            f"val loss={val_loss:.6e} | max raw grad norm={train_stats['max_grad_norm']:.6e} | "
            f"time={time.time() - start:.2f}s"
        )

        explosion_reason = None
        if not train_stats["valid"]:
            explosion_reason = f"invalid training epoch: {train_stats['failure']}"
        elif not val_stats["valid"] or not math.isfinite(val_loss):
            explosion_reason = f"invalid validation epoch: {val_stats['failure']}"
        elif math.isfinite(best_val) and val_loss > best_val * args.explosion_factor:
            explosion_reason = (
                f"validation loss {val_loss:.6e} exceeds best {best_val:.6e} "
                f"by more than {args.explosion_factor:g}x"
            )

        if explosion_reason is not None:
            print(f"Training explosion detected: {explosion_reason}")
            if args.disable_recovery:
                print("Recovery is disabled; stop training and retain the last valid best checkpoint.")
                break
            recoveries += 1
            if recoveries > args.max_recoveries:
                print(
                    f"Recovery limit exceeded ({args.max_recoveries}); stop training and retain the last valid best checkpoint."
                )
                break

            failed_lr = float(optimizer.param_groups[0]["lr"])
            if checkpoint_saved_this_run:
                rollback = load_checkpoint_file(ckpt_path, map_location=device)
                model.load_state_dict(rollback["model"])
                optimizer.load_state_dict(rollback["optimizer"])
                rollback_source = f"best epoch {rollback['epoch']}"
            else:
                model.load_state_dict(initial_model_state)
                optimizer.load_state_dict(initial_optimizer_state)
                rollback_source = "initial state"

            recovered_lr = max(failed_lr * args.lr_factor, args.min_lr)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = recovered_lr
            scheduler = make_plateau_scheduler(optimizer, args)
            bad_epochs = 0
            lr_reduced_on_plateau = False
            print(
                f"Recovery {recoveries}/{args.max_recoveries}: restored {rollback_source}; "
                f"lr {failed_lr:.3e} -> {recovered_lr:.3e}; early-stop window reset."
            )
            continue

        improved = val_loss < best_val - args.min_delta
        previous_lr = float(optimizer.param_groups[0]["lr"])
        # Once LR has been reduced for the current no-improvement streak, wait
        # for a new best instead of reducing it repeatedly before early stop.
        if scheduler is not None and (improved or not lr_reduced_on_plateau):
            scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        lr_reduced = current_lr < previous_lr
        if lr_reduced:
            lr_reduced_on_plateau = True
            print(f"🔻 LR reduced: {previous_lr:.3e} -> {current_lr:.3e}; no-improvement count preserved.")

        if improved:
            best_val = val_loss
            bad_epochs = 0
            lr_reduced_on_plateau = False
            payload = {
                "checkpoint_version": CHECKPOINT_VERSION,
                "training_protocol": TRAINING_PROTOCOL,
                "epoch": int(epoch),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "args": vars(args).copy(),
                "model_name": "TypedGraphPowerFlow",
                "model_config": expected_model_config(args, info),
                "best_val_loss": float(best_val),
            }
            atomic_save_checkpoint(payload, ckpt_path)
            checkpoint_saved_this_run = True
            print(f"🟩 Saved best checkpoint atomically: {ckpt_path}")
        else:
            bad_epochs += 1
            print(f"🔘 No improvement ({bad_epochs}/{early_patience}) | best val={best_val:.6e}")
            if not args.disable_early_stop and bad_epochs >= early_patience:
                print("Early stopping")
                break

    if not should_load_existing and not checkpoint_saved_this_run:
        raise RuntimeError(
            "Training did not produce a valid current-protocol checkpoint; refusing to fall back to an old/stale file."
        )
    print(f"Loading checkpoint : {ckpt_path}", flush=True)
    checkpoint = load_checkpoint_file(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    print(
        f"Checkpoint loaded  : OK (version={checkpoint.get('checkpoint_version', 'legacy')}, "
        f"protocol={checkpoint.get('training_protocol', 'missing')}, epoch={checkpoint.get('epoch', 'unknown')})",
        flush=True,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    test_start = time.perf_counter()
    print("Testing summary    : running full metrics", flush=True)
    metrics = evaluate_full_summary(model, test_loader, info, ybus, device, args.tol)
    if device.type == "cuda":
        torch.cuda.synchronize()
    test_elapsed = time.perf_counter() - test_start
    print_metric_summary(f"test metrics (tol={args.tol})", metrics)
    print(f"Test evaluation time             : {test_elapsed:.6f} s")

    print("Testing heatmap    : running cell metrics", flush=True)
    cells = evaluate_heatmap_cells(model, test_loader, info, ybus, device)
    print_heatmap_cell_summary(cells)
    print_region_mismatch_summary(cells, args.tol)
    heatmap_path = ckpt_path.parent / f"{data_system_tag(args.data_dir)}.csv"
    write_heatmap_cell_csv(cells, heatmap_path)

    print("Testing ill-conditioned: running success metrics", flush=True)
    ill_csv_path = ckpt_path.parent / f"{data_system_name(args.data_dir)}_ill.csv"
    ill_metrics = None
    if len(ill_test_set) > 0:
        ill_metrics = evaluate_full_summary(
            model,
            ill_test_loader,
            info,
            ybus,
            device,
            args.tol,
            nr_refine_max_steps=20,
            nr_refine_tol=1e-6,
            mismatch_csv_path=ill_csv_path,
        )
    print_ill_conditioned_summary(ill_metrics)
    if ill_metrics is not None:
        print(f"Ill-conditioned mismatch CSV saved: {ill_csv_path}")

if __name__ == "__main__":
    main()
