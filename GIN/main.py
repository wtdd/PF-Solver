import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import CommonPFDataset, collate_samples, load_system_info, load_ybus, move_batch, resolve_data_dir, split_train_val
from metrics import evaluate_full_summary, evaluate_heatmap_cells, print_heatmap_cell_summary, print_ill_conditioned_summary, print_metric_summary, print_region_mismatch_summary, GIN_loss, write_heatmap_cell_csv
from model import GlobalReceptiveGIN


ROOT = Path(__file__).resolve().parents[1]


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


def choose_log_path(args, ckpt_path: Path) -> Path:
    if args.load_trained_model and ckpt_path.exists():
        return ckpt_path.parent / f"{data_system_tag(args.data_dir)}_test.log"
    return ckpt_path.with_suffix(ckpt_path.suffix + ".log")


def make_plateau_scheduler(optimizer: torch.optim.Optimizer, args):
    if args.disable_scheduler:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.scheduler_patience,
        threshold=args.min_delta,
        threshold_mode="abs",
        min_lr=args.min_lr,
    )


def effective_early_stop_patience(args) -> int:
    if args.early_stop_patience is not None:
        return args.early_stop_patience
    return args.scheduler_patience * 2


def run_epoch(model, loader, optimizer, device, info, ybus, train: bool):
    model.train(train)
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            loss = GIN_loss(model, batch, info, ybus)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        n = int(batch["p_spec"].shape[0])
        total += float(loss.detach().item()) * n
        count += n
    return total / max(count, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=str(ROOT / "Data" / "IEEE_118"))
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--test_batch_size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler_patience", type=int, default=20)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--min_delta", type=float, default=1e-6)
    parser.add_argument("--disable_scheduler", action="store_true")
    parser.add_argument("--disable_early_stop", action="store_true")
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--ckpt", default="")
    parser.add_argument(
        "--load_trained_model",
        nargs="?",
        const=True,
        default=True,
        type=str2bool,
        help="If true and checkpoint exists, load it and only run test statistics; otherwise train first.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = load_system_info(args.data_dir)
    ybus = torch.tensor(load_ybus(args.data_dir), dtype=torch.complex64, device=device)
    train_all = CommonPFDataset(args.data_dir, "train")
    test_set = CommonPFDataset(args.data_dir, "test", exclude_source="ill-conditioned")
    ill_test_set = CommonPFDataset(args.data_dir, "test", source_filter="ill-conditioned", allow_empty=True)
    train_set, val_set = split_train_val(train_all, args.val_ratio, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_samples)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_samples)
    ill_test_loader = DataLoader(ill_test_set, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_samples)

    model = GlobalReceptiveGIN(info.state_dim, layers=args.layers).to(device)
    ckpt_dir = Path(__file__).resolve().parent / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.ckpt) if args.ckpt else ckpt_dir / f"{data_system_tag(args.data_dir)}_GIN_best.pt"
    ckpt_path = ckpt_path.expanduser().resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    start_log(choose_log_path(args, ckpt_path))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_plateau_scheduler(optimizer, args)
    early_patience = effective_early_stop_patience(args)

    print(f"Using device       : {device}")
    print(f"Data directory     : {resolve_data_dir(args.data_dir)}")
    print("Model              : Global-Receptive Graph Iteration Network")
    print(f"Checkpoint         : {ckpt_path}")
    print(f"Dataset sizes      : train={len(train_set)}, val={len(val_set)}, heatmap_test={len(test_set)}, ill_test={len(ill_test_set)}")
    print(f"Test batch size    : {args.test_batch_size}")
    print(f"Load trained model : {args.load_trained_model}")

    should_load_existing = bool(args.load_trained_model and ckpt_path.exists())
    if should_load_existing:
        print("Load checkpoint mode: skip training.")
        args.epochs = 0
    elif args.load_trained_model:
        print(f"Checkpoint not found for load_trained_model; start training: {ckpt_path}")

    best_val = math.inf
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, info, ybus, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, info, ybus, train=False)
        prev_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:04d}/{args.epochs} | lr={current_lr:.3e} | train loss={train_loss:.6e} | val loss={val_loss:.6e} | time={time.time() - start:.2f}s")
        if current_lr < prev_lr:
            print(f"🔻 LR reduced: {prev_lr:.3e} -> {current_lr:.3e}")
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print(f"🟩 saved best checkpoint: {ckpt_path}")
        else:
            bad_epochs += 1
            print(f"🔘 no improvement ({bad_epochs}/{early_patience}) | best val={best_val:.6e}")
            if (not args.disable_early_stop) and bad_epochs >= early_patience:
                print(" early stopping")
                break

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found after training/load decision: {ckpt_path}")
    print(f"Loading checkpoint : {ckpt_path}", flush=True)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    print("Checkpoint loaded  : OK", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    print("Testing summary    : running full metrics", flush=True)
    metrics = evaluate_full_summary(model, test_loader, info, ybus, device, args.tol)
    if device.type == "cuda":
        torch.cuda.synchronize()
    test_elapsed = time.perf_counter() - t0

    print_metric_summary(f"test metrics (tol={args.tol})", metrics)
    print(f"Test evaluation time             : {test_elapsed:.6f} s")
    print("Testing heatmap    : running cell metrics", flush=True)
    cells = evaluate_heatmap_cells(model, test_loader, info, ybus, device)
    print_heatmap_cell_summary(cells)
    print_region_mismatch_summary(cells, args.tol)
    csv_result = data_system_tag(args.data_dir) + ".csv"
    write_heatmap_cell_csv(cells, ckpt_path.parent / csv_result)
    print("Testing ill-conditioned: running success metrics", flush=True)
    ill_csv_path = ckpt_path.parent / f"{data_system_name(args.data_dir)}_ill.csv"
    ill_metrics = evaluate_full_summary(model, ill_test_loader, info, ybus, device, args.tol, nr_refine_max_steps=20, nr_refine_tol=1e-6, mismatch_csv_path=ill_csv_path) if len(ill_test_set) > 0 else None
    print_ill_conditioned_summary(ill_metrics)
    if ill_metrics is not None:
        print(f"Ill-conditioned mismatch CSV saved: {ill_csv_path}")


if __name__ == "__main__":
    main()
