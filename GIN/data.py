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

    @property
    def state_dim(self) -> int:
        return len(self.non_slack_idx) + len(self.pq_idx)


def load_system_info(data_dir: str) -> PFSystemInfo:
    root = resolve_data_dir(data_dir)
    bus_static = pd.read_csv(root / "bus_static.csv").sort_values("bus0")
    n_bus = int(bus_static["bus0"].max()) + 1
    slack_idx = int(bus_static.loc[bus_static["is_slack"].astype(int) == 1, "bus0"].iloc[0])
    pv_idx = bus_static.loc[bus_static["is_pv"].astype(int) == 1, "bus0"].astype(int).tolist()
    pq_idx = bus_static.loc[bus_static["is_pq"].astype(int) == 1, "bus0"].astype(int).tolist()
    non_slack_idx = [i for i in range(n_bus) if i != slack_idx]
    return PFSystemInfo(n_bus=n_bus, slack_idx=slack_idx, pv_idx=pv_idx, pq_idx=pq_idx, non_slack_idx=non_slack_idx)


def load_ybus(data_dir: str) -> np.ndarray:
    root = resolve_data_dir(data_dir)
    info = load_system_info(str(root))
    ybus = np.zeros((info.n_bus, info.n_bus), dtype=np.complex64)
    ybus_df = pd.read_csv(root / "ybus.csv")
    for row in ybus_df.itertuples(index=False):
        ybus[int(row.from_bus0), int(row.to_bus0)] = np.complex64(float(row.g) + 1j * float(row.b))
    return ybus


class CommonPFDataset(Dataset):
    """Data_Gen2 CSV loader for GIN.

    The paper's GIN layer uses the current operating state and power injections.
    This loader keeps the start state, target state, and specified injections so
    model.py can build Eq. (15)-(17) graph-iteration updates.
    """

    def __init__(self, data_dir: str, split: str, source_filter: str = None, exclude_source: str = None, allow_empty: bool = False):
        self.root = resolve_data_dir(data_dir)
        self.info = load_system_info(str(self.root))
        meta = pd.read_csv(self.root / "meta.csv")
        meta = meta[meta["split"].astype(str).str.lower() == split.lower()].copy()
        if source_filter is not None:
            meta = meta[meta["source"].astype(str).str.lower() == source_filter.lower()].copy()
        if exclude_source is not None:
            meta = meta[meta["source"].astype(str).str.lower() != exclude_source.lower()].copy()
        meta = meta.sort_values("sample_id").reset_index(drop=True)
        if len(meta) == 0:
            if allow_empty:
                self.samples = []
                return
            raise ValueError(f"No samples with split='{split}' in {self.root / 'meta.csv'}")
        meta_by_id = {int(row.sample_id): row for row in meta.itertuples(index=False)}
        bus_state = pd.read_csv(self.root / "bus_state.csv")
        keep_ids = set(meta["sample_id"].astype(int).tolist())
        bus_state = bus_state[bus_state["sample_id"].astype(int).isin(keep_ids)].copy()
        self.samples = []
        for sample_id, group in bus_state.groupby("sample_id", sort=True):
            sample_id = int(sample_id)
            group = group.sort_values("bus0")
            if len(group) != self.info.n_bus:
                raise ValueError(f"sample_id={sample_id} has {len(group)} rows, expected {self.info.n_bus}")
            meta_row = meta_by_id[sample_id]
            source = str(getattr(meta_row, "source", ""))

            def col(name: str, fallback: str = None) -> torch.Tensor:
                source = fallback if name not in group.columns and fallback is not None else name
                return torch.tensor(group[source].to_numpy(dtype=np.float32), dtype=torch.float32)

            self.samples.append(
                {
                    "sample_id": torch.tensor(sample_id, dtype=torch.long),
                    "p_spec": col("p_spec_pu"),
                    "q_spec": col("q_spec_pu"),
                    "vm_start": col("vm_start"),
                    "va_start": col("va_start_rad"),
                    "vm_true": col("vm_true"),
                    "va_true": col("va_true_rad"),
                    "level": torch.tensor(
                        [
                            float(getattr(meta_row, "x_low", 0.0)),
                            float(getattr(meta_row, "x_high", 0.0)),
                            float(getattr(meta_row, "pq_low", 0.0)),
                            float(getattr(meta_row, "pq_high", 0.0)),
                            float(getattr(meta_row, "x_signed", 0.0)),
                            float(getattr(meta_row, "pq_signed", 0.0)),
                        ],
                        dtype=torch.float32,
                    ),
                    "valid_label": torch.tensor(int(getattr(meta_row, "valid_label", 1)), dtype=torch.long),
                    "ill_conditioned": torch.tensor(1 if source.strip().lower() == "ill-conditioned" else 0, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


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
        sin_np = torch.sin(th_np)
        cos_np = torch.cos(th_np)
        g_np = g[non][:, pq]
        b_np = b[non][:, pq]
        n_block = vm_non.unsqueeze(2) * (g_np.unsqueeze(0) * cos_np + b_np.unsqueeze(0) * sin_np)
        row_for_pq = torch.tensor([info.non_slack_idx.index(int(bus)) for bus in info.pq_idx], dtype=torch.long, device=device)
        d_pq = torch.arange(n_pq, device=device)
        n_block[:, row_for_pq, d_pq] = p_calc[:, pq] / vm_pq.clamp_min(1e-8) + torch.diagonal(g[pq][:, pq]).unsqueeze(0) * vm_pq
        jmat[:, :n_non, n_non:] = n_block

        th_pn = va_pq.unsqueeze(2) - va_non.unsqueeze(1)
        sin_pn = torch.sin(th_pn)
        cos_pn = torch.cos(th_pn)
        g_pn = g[pq][:, non]
        b_pn = b[pq][:, non]
        m_block = -vm_pq.unsqueeze(2) * vm_non.unsqueeze(1) * (g_pn.unsqueeze(0) * cos_pn + b_pn.unsqueeze(0) * sin_pn)
        m_block[:, d_pq, row_for_pq] = p_calc[:, pq] - torch.diagonal(g[pq][:, pq]).unsqueeze(0) * vm_pq.square()
        jmat[:, n_non:, :n_non] = m_block

        th_pp = va_pq.unsqueeze(2) - va_pq.unsqueeze(1)
        sin_pp = torch.sin(th_pp)
        cos_pp = torch.cos(th_pp)
        g_pp = g[pq][:, pq]
        b_pp = b[pq][:, pq]
        l_block = vm_pq.unsqueeze(2) * (g_pp.unsqueeze(0) * sin_pp - b_pp.unsqueeze(0) * cos_pp)
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
    for x_low, x_high, pq_low, pq_high in cells.keys():
        stats = cells[(x_low, x_high, pq_low, pq_high)]
        x_center = round((float(x_low) + float(x_high)) / 2.0, 10)
        y_center = round((float(pq_low) + float(pq_high)) / 2.0, 10)
        value_by_xy[(x_center, y_center)] = max(float(stats["p_max"]), float(stats["q_max"]))
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["PQ\\X"] + [f"{x:.1f}" for x in x_values])
        for y in y_values:
            writer.writerow([f"{y:.1f}"] + [value_by_xy.get((x, y), "") for x in x_values])
    print(f"Heatmap CSV saved: {output_csv}")
