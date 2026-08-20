import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


def resolve_data_dir(data_dir: str) -> Path:
    root = Path(data_dir).expanduser().resolve()
    common = root / "Common"
    return common if common.exists() else root


@dataclass
class PFSystemInfo:
    n_bus: int
    slack_idx: int
    pv_idx: List[int]
    pq_idx: List[int]
    non_slack_idx: List[int]
    branch_from: List[int]
    branch_to: List[int]
    branch_features: np.ndarray

    @property
    def state_dim(self) -> int:
        return len(self.non_slack_idx) + len(self.pq_idx)

    @property
    def n_branch(self) -> int:
        return len(self.branch_from)


def load_system_info(data_dir: str) -> PFSystemInfo:
    root = resolve_data_dir(data_dir)
    bus_static = pd.read_csv(root / "bus_static.csv").sort_values("bus0")
    n_bus = int(bus_static["bus0"].max()) + 1
    slack_rows = bus_static.loc[bus_static["is_slack"].astype(int) == 1, "bus0"]
    if len(slack_rows) != 1:
        raise ValueError(f"Expected exactly one slack bus in {root / 'bus_static.csv'}, got {len(slack_rows)}")
    slack_idx = int(slack_rows.iloc[0])
    pv_idx = bus_static.loc[bus_static["is_pv"].astype(int) == 1, "bus0"].astype(int).tolist()
    pq_idx = bus_static.loc[bus_static["is_pq"].astype(int) == 1, "bus0"].astype(int).tolist()
    non_slack_idx = [i for i in range(n_bus) if i != slack_idx]

    branch_path = root / "branch_static.csv"
    branch = pd.read_csv(branch_path).sort_values("branch_id")
    if "status" in branch.columns:
        branch = branch[branch["status"].astype(float) != 0.0].copy()
    r = branch["r"].to_numpy(dtype=np.float32)
    x = branch["x"].to_numpy(dtype=np.float32)
    b = branch["b"].to_numpy(dtype=np.float32)
    impedance = np.sqrt(r * r + x * x)
    rho = np.divide(1.0, impedance, out=np.zeros_like(impedance), where=impedance > 1e-12)
    delta = -np.arctan2(x, r).astype(np.float32)
    branch_features = np.stack([rho, delta, b], axis=1).astype(np.float32, copy=False)
    return PFSystemInfo(
        n_bus=n_bus,
        slack_idx=slack_idx,
        pv_idx=pv_idx,
        pq_idx=pq_idx,
        non_slack_idx=non_slack_idx,
        branch_from=branch["from_bus0"].astype(int).tolist(),
        branch_to=branch["to_bus0"].astype(int).tolist(),
        branch_features=branch_features,
    )


def load_ybus(data_dir: str) -> np.ndarray:
    root = resolve_data_dir(data_dir)
    info = load_system_info(str(root))
    ybus = np.zeros((info.n_bus, info.n_bus), dtype=np.complex64)
    ybus_df = pd.read_csv(root / "ybus.csv", usecols=["from_bus0", "to_bus0", "g", "b"])
    rows = ybus_df["from_bus0"].to_numpy(dtype=np.int64)
    cols = ybus_df["to_bus0"].to_numpy(dtype=np.int64)
    vals = ybus_df["g"].to_numpy(dtype=np.float32) + 1j * ybus_df["b"].to_numpy(dtype=np.float32)
    ybus[rows, cols] = vals.astype(np.complex64)
    return ybus


_DENSE_CACHE: Dict[str, Dict[str, object]] = {}


def _load_dense_cache(root: Path, info: PFSystemInfo) -> Dict[str, object]:
    """Read the large bus_state CSV once and store compact [sample, bus] tensors."""
    key = str(root)
    if key in _DENSE_CACHE:
        return _DENSE_CACHE[key]

    columns = [
        "sample_id", "split", "bus0", "p_spec_pu", "q_spec_pu",
        "vm_start", "va_start_rad", "vm_true", "va_true_rad",
    ]
    frame = pd.read_csv(
        root / "bus_state.csv",
        usecols=columns,
        dtype={
            "sample_id": np.int64,
            "split": "category",
            "bus0": np.int32,
            "p_spec_pu": np.float32,
            "q_spec_pu": np.float32,
            "vm_start": np.float32,
            "va_start_rad": np.float32,
            "vm_true": np.float32,
            "va_true_rad": np.float32,
        },
    )
    frame["split"] = frame["split"].astype(str).str.lower()
    frame.sort_values(["split", "sample_id", "bus0"], inplace=True, ignore_index=True)
    if len(frame) % info.n_bus != 0:
        raise ValueError(f"{root / 'bus_state.csv'} row count is not divisible by n_bus={info.n_bus}")
    n_samples = len(frame) // info.n_bus
    bus_grid = frame["bus0"].to_numpy(dtype=np.int64).reshape(n_samples, info.n_bus)
    expected = np.arange(info.n_bus, dtype=np.int64)[None, :]
    if not np.all(bus_grid == expected):
        raise ValueError("Every (split, sample_id) block in bus_state.csv must contain bus0=0..n_bus-1 exactly once")

    split_grid = frame["split"].to_numpy().reshape(n_samples, info.n_bus)
    id_grid = frame["sample_id"].to_numpy(dtype=np.int64).reshape(n_samples, info.n_bus)
    keys = [(str(split_grid[i, 0]), int(id_grid[i, 0])) for i in range(n_samples)]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate (split, sample_id) blocks found in bus_state.csv")
    key_to_row = {sample_key: i for i, sample_key in enumerate(keys)}

    tensor_columns = {}
    for output_name, csv_name in (
        ("p_spec", "p_spec_pu"),
        ("q_spec", "q_spec_pu"),
        ("vm_start", "vm_start"),
        ("va_start", "va_start_rad"),
        ("vm_true", "vm_true"),
        ("va_true", "va_true_rad"),
    ):
        values = frame[csv_name].to_numpy(dtype=np.float32).reshape(n_samples, info.n_bus)
        tensor_columns[output_name] = torch.from_numpy(values.copy())
    cache = {"key_to_row": key_to_row, **tensor_columns}
    _DENSE_CACHE[key] = cache
    return cache


class CommonPFDataset(Dataset):
    """Compact loader for the CSV schema emitted by Slover_Gen_Data.cpp."""

    def __init__(self, data_dir: str, split: str, source_filter: str = None, exclude_source: str = None, allow_empty: bool = False):
        self.root = resolve_data_dir(data_dir)
        self.info = load_system_info(str(self.root))
        meta = pd.read_csv(self.root / "meta.csv")
        meta["split"] = meta["split"].astype(str).str.lower()
        meta = meta[meta["split"] == split.lower()].copy()
        if source_filter is not None:
            meta = meta[meta["source"].astype(str).str.lower() == source_filter.lower()].copy()
        if exclude_source is not None:
            meta = meta[meta["source"].astype(str).str.lower() != exclude_source.lower()].copy()
        meta.sort_values("sample_id", inplace=True, ignore_index=True)
        if len(meta) == 0 and not allow_empty:
            raise ValueError(f"No samples with split='{split}' in {self.root / 'meta.csv'}")

        cache = _load_dense_cache(self.root, self.info)
        key_to_row = cache["key_to_row"]
        self.cache = cache
        self.rows: List[int] = []
        self.sample_ids: List[int] = []
        levels = []
        valid_labels = []
        ill_flags = []
        for row in meta.itertuples(index=False):
            sample_id = int(row.sample_id)
            sample_key = (split.lower(), sample_id)
            if sample_key not in key_to_row:
                raise ValueError(f"sample_id={sample_id}, split={split} exists in meta.csv but not bus_state.csv")
            self.rows.append(int(key_to_row[sample_key]))
            self.sample_ids.append(sample_id)
            levels.append([
                float(getattr(row, "x_low", 0.0)), float(getattr(row, "x_high", 0.0)),
                float(getattr(row, "pq_low", 0.0)), float(getattr(row, "pq_high", 0.0)),
                float(getattr(row, "x_signed", 0.0)), float(getattr(row, "pq_signed", 0.0)),
            ])
            valid_labels.append(int(getattr(row, "valid_label", 1)))
            source = str(getattr(row, "source", "")).strip().lower()
            ill_flags.append(1 if source == "ill-conditioned" else 0)
        self.levels = torch.tensor(levels, dtype=torch.float32) if levels else torch.empty((0, 6), dtype=torch.float32)
        self.valid_labels = torch.tensor(valid_labels, dtype=torch.long)
        self.ill_flags = torch.tensor(ill_flags, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        row = self.rows[item]
        return {
            "sample_id": torch.tensor(self.sample_ids[item], dtype=torch.long),
            "p_spec": self.cache["p_spec"][row],
            "q_spec": self.cache["q_spec"][row],
            "vm_start": self.cache["vm_start"][row],
            "va_start": self.cache["va_start"][row],
            "vm_true": self.cache["vm_true"][row],
            "va_true": self.cache["va_true"][row],
            "level": self.levels[item],
            "valid_label": self.valid_labels[item],
            "ill_conditioned": self.ill_flags[item],
        }


def collate_samples(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0].keys()}


def split_train_val(dataset: Dataset, val_ratio: float, seed: int) -> Tuple[Subset, Subset]:
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_val = int(round(len(indices) * val_ratio))
    if len(indices) > 1:
        n_val = min(max(n_val, 1), len(indices) - 1)
    else:
        n_val = 0
    return Subset(dataset, indices[n_val:].tolist()), Subset(dataset, indices[:n_val].tolist())


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def idx(values: List[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long, device=device)


def voltage_to_complex(vm: torch.Tensor, va: torch.Tensor) -> torch.Tensor:
    return vm.to(torch.complex64) * torch.exp(1j * va.to(torch.complex64))


def calc_power(ybus: torch.Tensor, vm: torch.Tensor, va: torch.Tensor):
    vc = voltage_to_complex(vm, va)
    ic = torch.matmul(ybus, vc.unsqueeze(-1)).squeeze(-1)
    sc = vc * ic.conj()
    return sc.real.to(vm.dtype), sc.imag.to(vm.dtype)


def mismatch_from_voltage(ybus: torch.Tensor, p_spec: torch.Tensor, q_spec: torch.Tensor, vm: torch.Tensor, va: torch.Tensor):
    p_calc, q_calc = calc_power(ybus, vm, va)
    return p_spec - p_calc, q_spec - q_calc


def polar_jacobian(ybus: torch.Tensor, vm: torch.Tensor, va: torch.Tensor, info: PFSystemInfo) -> torch.Tensor:
    dtype = vm.dtype
    device = vm.device
    g = ybus.real.to(dtype)
    b = ybus.imag.to(dtype)
    p_calc, q_calc = calc_power(ybus, vm, va)
    non = idx(info.non_slack_idx, device)
    pq = idx(info.pq_idx, device)
    n_non = int(non.numel())
    n_pq = int(pq.numel())
    state_dim = n_non + n_pq
    jmat = torch.empty(vm.shape[0], state_dim, state_dim, dtype=dtype, device=device)

    vm_non = vm[:, non]
    va_non = va[:, non]
    th_nn = va_non.unsqueeze(2) - va_non.unsqueeze(1)
    sin_nn = torch.sin(th_nn)
    cos_nn = torch.cos(th_nn)
    g_nn = g[non][:, non]
    b_nn = b[non][:, non]
    h = vm_non.unsqueeze(2) * vm_non.unsqueeze(1) * (g_nn.unsqueeze(0) * sin_nn - b_nn.unsqueeze(0) * cos_nn)
    d_non = torch.arange(n_non, device=device)
    h[:, d_non, d_non] = -q_calc[:, non] - torch.diagonal(b_nn).unsqueeze(0) * vm_non.square()
    jmat[:, :n_non, :n_non] = h

    if n_pq:
        vm_pq = vm[:, pq]
        va_pq = va[:, pq]
        th_np = va_non.unsqueeze(2) - va_pq.unsqueeze(1)
        g_np = g[non][:, pq]
        b_np = b[non][:, pq]
        n_block = vm_non.unsqueeze(2) * (g_np.unsqueeze(0) * torch.cos(th_np) + b_np.unsqueeze(0) * torch.sin(th_np))
        row_for_pq = torch.tensor([info.non_slack_idx.index(int(bus)) for bus in info.pq_idx], dtype=torch.long, device=device)
        d_pq = torch.arange(n_pq, device=device)
        n_block[:, row_for_pq, d_pq] = p_calc[:, pq] / vm_pq.clamp_min(1e-8) + torch.diagonal(g[pq][:, pq]).unsqueeze(0) * vm_pq
        jmat[:, :n_non, n_non:] = n_block

        th_pn = va_pq.unsqueeze(2) - va_non.unsqueeze(1)
        g_pn = g[pq][:, non]
        b_pn = b[pq][:, non]
        m_block = -vm_pq.unsqueeze(2) * vm_non.unsqueeze(1) * (g_pn.unsqueeze(0) * torch.cos(th_pn) + b_pn.unsqueeze(0) * torch.sin(th_pn))
        m_block[:, d_pq, row_for_pq] = p_calc[:, pq] - torch.diagonal(g[pq][:, pq]).unsqueeze(0) * vm_pq.square()
        jmat[:, n_non:, :n_non] = m_block

        th_pp = va_pq.unsqueeze(2) - va_pq.unsqueeze(1)
        g_pp = g[pq][:, pq]
        b_pp = b[pq][:, pq]
        l_block = vm_pq.unsqueeze(2) * (g_pp.unsqueeze(0) * torch.sin(th_pp) - b_pp.unsqueeze(0) * torch.cos(th_pp))
        l_block[:, d_pq, d_pq] = q_calc[:, pq] / vm_pq.clamp_min(1e-8) - torch.diagonal(b_pp).unsqueeze(0) * vm_pq
        jmat[:, n_non:, n_non:] = l_block
    return jmat


def pack_state(vm: torch.Tensor, va: torch.Tensor, info: PFSystemInfo) -> torch.Tensor:
    device = vm.device
    return torch.cat([va[:, idx(info.non_slack_idx, device)], vm[:, idx(info.pq_idx, device)]], dim=1)


def unpack_state(state: torch.Tensor, batch: Dict[str, torch.Tensor], info: PFSystemInfo):
    device = state.device
    vm = batch.get("vm_start", batch["vm_true"]).clone()
    va = batch.get("va_start", batch["va_true"]).clone()
    n_non = len(info.non_slack_idx)
    va[:, idx(info.non_slack_idx, device)] = state[:, :n_non]
    vm[:, idx(info.pq_idx, device)] = state[:, n_non:]
    return torch.clamp(vm, min=0.2, max=2.0), va


def cell_key(level_row: torch.Tensor) -> Tuple[float, float, float, float]:
    vals = level_row.detach().cpu().tolist()
    return tuple(0.0 if abs(float(v)) < 5e-8 else float(v) for v in vals[:4])


def write_heatmap_csv(cells: Dict[Tuple[float, float, float, float], Dict[str, float]], output_csv) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    x_values = sorted({round((float(key[0]) + float(key[1])) / 2.0, 10) for key in cells.keys()})
    y_values = sorted({round((float(key[2]) + float(key[3])) / 2.0, 10) for key in cells.keys()})
    value_by_xy = {}
    for key, stats in cells.items():
        x_center = round((float(key[0]) + float(key[1])) / 2.0, 10)
        y_center = round((float(key[2]) + float(key[3])) / 2.0, 10)
        value_by_xy[(x_center, y_center)] = max(float(stats["p_max"]), float(stats["q_max"]))
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["PQ\\X"] + [f"{x:.1f}" for x in x_values])
        for y in y_values:
            writer.writerow([f"{y:.1f}"] + [value_by_xy.get((x, y), "") for x in x_values])
    print(f"Heatmap CSV saved: {output_csv}")
