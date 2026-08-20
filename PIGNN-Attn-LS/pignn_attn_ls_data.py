import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


_CACHE_VERSION = 3
_STORE_CACHE: Dict[str, "_CommonTensorStore"] = {}


def resolve_data_dir(data_dir: str) -> Path:
    root = Path(data_dir).expanduser().resolve()
    common = root / "Common"
    return common if common.exists() else root


def data_system_name(data_dir: str) -> str:
    root = resolve_data_dir(data_dir)
    return root.parent.name if root.name.lower() == "common" else root.name


def load_ybus(data_dir: str) -> np.ndarray:
    root = resolve_data_dir(data_dir)
    bus_static = pd.read_csv(root / "bus_static.csv", usecols=["bus0"])
    n_bus = int(bus_static["bus0"].max()) + 1
    ybus = np.zeros((n_bus, n_bus), dtype=np.complex64)
    ybus_df = pd.read_csv(root / "ybus.csv", usecols=["from_bus0", "to_bus0", "g", "b"])
    i = ybus_df["from_bus0"].to_numpy(dtype=np.int64)
    j = ybus_df["to_bus0"].to_numpy(dtype=np.int64)
    ybus[i, j] = (
        ybus_df["g"].to_numpy(dtype=np.float32)
        + 1j * ybus_df["b"].to_numpy(dtype=np.float32)
    ).astype(np.complex64)
    return ybus


def build_upper_line_features(root: Path, ybus: np.ndarray, eps: float = 1e-12):
    """Build the paper's [series conductance, susceptance, shunt] edge inputs.

    The network equations always use the exact exported Ybus.  branch_static.csv
    is used only for the edge-aware attention feature (half charging per end).
    """
    n_bus = int(ybus.shape[0])
    pairs = np.stack(np.triu_indices(n_bus, k=1), axis=1)
    yij = ybus[pairs[:, 0], pairs[:, 1]]
    yji = ybus[pairs[:, 1], pairs[:, 0]]
    connected = (np.abs(yij) > eps) | (np.abs(yji) > eps)
    y_series = np.zeros(len(pairs), dtype=np.complex64)
    both = (np.abs(yij) > eps) & (np.abs(yji) > eps)
    y_series[both] = -0.5 * (yij[both] + yji[both])
    only_ij = (np.abs(yij) > eps) & ~both
    only_ji = (np.abs(yji) > eps) & ~both
    y_series[only_ij] = -yij[only_ij]
    y_series[only_ji] = -yji[only_ji]

    y_charging = np.zeros(len(pairs), dtype=np.float32)
    branch_path = root / "branch_static.csv"
    if branch_path.exists():
        branch = pd.read_csv(
            branch_path,
            usecols=["from_bus0", "to_bus0", "b", "status"],
        )
        pair_to_idx = {(int(i), int(j)): k for k, (i, j) in enumerate(pairs)}
        for row in branch.itertuples(index=False):
            if int(row.status) == 0:
                continue
            i, j = sorted((int(row.from_bus0), int(row.to_bus0)))
            idx = pair_to_idx.get((i, j))
            if idx is not None:
                y_charging[idx] += np.float32(0.5 * float(row.b))
    return connected.astype(np.bool_), y_series, y_charging


def common_type_to_sota_type(common_type: np.ndarray) -> np.ndarray:
    # Common CSV: 3=slack, 2=PV, 1=PQ. PIGNN-Attn-LS: 1=slack, 2=PV, 3=PQ.
    out = np.full(common_type.shape, 3, dtype=np.int64)
    out[common_type == 3] = 1
    out[common_type == 2] = 2
    return out


def _source_signature(root: Path) -> Tuple[Tuple[str, int, int], ...]:
    signature = []
    for name in ("meta.csv", "bus_state.csv", "bus_static.csv", "branch_static.csv", "ybus.csv"):
        path = root / name
        if path.exists():
            stat = path.stat()
            signature.append((name, int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(signature)


class _CommonTensorStore:
    """One compact in-memory representation shared by train/test datasets.

    Only the eight CSV columns PIGNN-Attn-LS consumes are read. Static grid
    tensors are stored once instead of once per sample.  A validated tensor
    cache makes subsequent launches independent of multi-gigabyte CSV parsing.
    """

    def __init__(self, root: Path, rebuild_cache: bool = False):
        self.root = root
        cache_dir = root / "processed_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / f"pignn_attn_ls_common_v{_CACHE_VERSION}.pt"
        signature = _source_signature(root)
        payload = None
        if self.cache_path.exists() and not rebuild_cache:
            try:
                candidate = torch.load(self.cache_path, map_location="cpu", weights_only=False)
                if candidate.get("version") == _CACHE_VERSION and tuple(candidate.get("signature", ())) == signature:
                    payload = candidate
                    print(f"Data cache          : {self.cache_path.name}")
            except (OSError, RuntimeError, EOFError, ValueError):
                payload = None
        if payload is None:
            payload = self._build(signature)
            tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
            torch.save(payload, tmp)
            tmp.replace(self.cache_path)
            print(f"Data cached         : {self.cache_path.name}")

        for key, value in payload.items():
            if key not in ("version", "signature"):
                setattr(self, key, value)

    def _build(self, signature):
        print(f"Building PIGNN-Attn-LS compact cache from: {self.root.name}")
        meta_columns = [
            "sample_id", "split", "source", "is_pf_from_start_converged", "valid_label",
            "x_low", "x_high", "pq_low", "pq_high", "x_signed", "pq_signed",
        ]
        meta = pd.read_csv(self.root / "meta.csv", usecols=meta_columns)
        meta = meta.sort_values("sample_id").reset_index(drop=True)
        if meta["sample_id"].duplicated().any():
            raise ValueError("meta.csv contains duplicate sample_id values")

        bus_static = pd.read_csv(self.root / "bus_static.csv", usecols=["bus0", "type"]).sort_values("bus0")
        n_bus = int(bus_static["bus0"].max()) + 1
        if len(bus_static) != n_bus:
            raise ValueError(f"bus_static.csv has {len(bus_static)} rows, expected {n_bus}")

        state_columns = [
            "sample_id", "bus0", "p_spec_pu", "q_spec_pu", "vm_start",
            "va_start_rad", "vm_true", "va_true_rad",
        ]
        dtypes = {
            "sample_id": np.int64, "bus0": np.int16,
            "p_spec_pu": np.float32, "q_spec_pu": np.float32,
            "vm_start": np.float32, "va_start_rad": np.float32,
            "vm_true": np.float32, "va_true_rad": np.float32,
        }
        state = pd.read_csv(self.root / "bus_state.csv", usecols=state_columns, dtype=dtypes)
        expected_rows = len(meta) * n_bus
        if len(state) != expected_rows:
            raise ValueError(f"bus_state.csv has {len(state)} rows, expected {expected_rows}")
        sample_values = state["sample_id"].to_numpy(dtype=np.int64, copy=False)
        bus_values = state["bus0"].to_numpy(dtype=np.int64, copy=False)
        ordered = (
            np.all(sample_values[:-1] <= sample_values[1:])
            and np.array_equal(bus_values.reshape(-1, n_bus)[0], np.arange(n_bus))
        )
        if not ordered:
            state = state.sort_values(["sample_id", "bus0"]).reset_index(drop=True)
            sample_values = state["sample_id"].to_numpy(dtype=np.int64, copy=False)
            bus_values = state["bus0"].to_numpy(dtype=np.int64, copy=False)
        state_ids = sample_values.reshape(-1, n_bus)[:, 0]
        if not np.all(sample_values.reshape(-1, n_bus) == state_ids[:, None]):
            raise ValueError("bus_state.csv sample rows are incomplete or interleaved")
        if not np.all(bus_values.reshape(-1, n_bus) == np.arange(n_bus)[None, :]):
            raise ValueError("bus_state.csv bus0 rows are incomplete or out of order")
        meta = meta.set_index("sample_id").loc[state_ids].reset_index()

        def matrix(column: str) -> torch.Tensor:
            arr = state[column].to_numpy(dtype=np.float32, copy=True).reshape(-1, n_bus)
            return torch.from_numpy(arr)

        p_spec, q_spec = matrix("p_spec_pu"), matrix("q_spec_pu")
        vm_start, va_start = matrix("vm_start"), matrix("va_start_rad")
        vm_true, va_true = matrix("vm_true"), matrix("va_true_rad")
        ybus_np = load_ybus(str(self.root))
        line_np, yline_np, yc_np = build_upper_line_features(self.root, ybus_np)

        return {
            "version": _CACHE_VERSION,
            "signature": signature,
            "sample_id": torch.from_numpy(state_ids.copy()),
            "split": meta["split"].astype(str).str.lower().tolist(),
            "source": meta["source"].fillna("").astype(str).str.lower().tolist(),
            "valid_label": torch.from_numpy(meta["valid_label"].fillna(0).to_numpy(dtype=np.int64)),
            "is_pf_from_start_converged": torch.from_numpy(meta["is_pf_from_start_converged"].fillna(0).to_numpy(dtype=np.int64)),
            "level": torch.from_numpy(meta[["x_low", "x_high", "pq_low", "pq_high", "x_signed", "pq_signed"]].fillna(0.0).to_numpy(dtype=np.float32)),
            "S_start": torch.complex(p_spec, q_spec),
            "V_start": torch.stack((vm_start, va_start), dim=-1),
            "V_newton": torch.stack((vm_true, va_true), dim=-1),
            "bus_type": torch.from_numpy(common_type_to_sota_type(bus_static["type"].to_numpy(dtype=np.int64))),
            "Lines_connected": torch.from_numpy(line_np),
            "Y_Lines": torch.from_numpy(yline_np),
            "Y_C_Lines": torch.from_numpy(yc_np),
            "Ybus": torch.from_numpy(ybus_np),
            "n_bus": n_bus,
        }


def _get_store(data_dir: str, rebuild_cache: bool = False) -> _CommonTensorStore:
    root = resolve_data_dir(data_dir)
    key = str(root).lower()
    if rebuild_cache or key not in _STORE_CACHE:
        _STORE_CACHE[key] = _CommonTensorStore(root, rebuild_cache=rebuild_cache)
    return _STORE_CACHE[key]


class CommonCSVPowerFlowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        source_filter: Optional[str] = None,
        exclude_source: Optional[str] = None,
        allow_empty: bool = False,
        rebuild_cache: bool = False,
    ):
        self.root = resolve_data_dir(data_dir)
        self.store = _get_store(data_dir, rebuild_cache=rebuild_cache)
        split_value = split.strip().lower()
        source_value = source_filter.strip().lower() if source_filter else None
        exclude_value = exclude_source.strip().lower() if exclude_source else None
        indices = []
        for idx, (row_split, row_source) in enumerate(zip(self.store.split, self.store.source)):
            if row_split != split_value:
                continue
            if source_value is not None and row_source != source_value:
                continue
            if exclude_value is not None and row_source == exclude_value:
                continue
            indices.append(idx)
        self.indices = torch.tensor(indices, dtype=torch.long)
        if not indices and not allow_empty:
            raise ValueError(f"No samples with split='{split}' and requested source filter in {self.root / 'meta.csv'}")

    @property
    def topology(self) -> Dict[str, torch.Tensor]:
        return {
            "bus_type": self.store.bus_type,
            "Lines_connected": self.store.Lines_connected,
            "Y_Lines": self.store.Y_Lines,
            "Y_C_Lines": self.store.Y_C_Lines,
            "Ybus": self.store.Ybus,
        }

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = int(self.indices[idx])
        return {
            "sample_id": self.store.sample_id[row],
            "valid_label": self.store.valid_label[row],
            "is_pf_from_start_converged": self.store.is_pf_from_start_converged[row],
            "level": self.store.level[row],
            "S_start": self.store.S_start[row],
            "V_start": self.store.V_start[row],
            "V_newton": self.store.V_newton[row],
        }


def _base_dataset(dataset: Dataset) -> CommonCSVPowerFlowDataset:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, CommonCSVPowerFlowDataset):
        raise TypeError("PIGNN-Attn-LS loader requires CommonCSVPowerFlowDataset or Subset")
    return dataset


class SharedTopologyCollator:
    def __init__(self, dataset: Dataset, blockdiag: bool = False):
        self.topology = _base_dataset(dataset).topology
        self.blockdiag = blockdiag

    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        dynamic = {key: torch.stack([sample[key] for sample in samples], dim=0) for key in samples[0]}
        batch_size = len(samples)
        if not self.blockdiag:
            # pin_memory cannot pin a zero-stride expanded view; this small
            # integer mask is the only static tensor repeated per sample.
            dynamic["bus_type"] = self.topology["bus_type"].unsqueeze(0).repeat(batch_size, 1)
            # These tensors describe one topology shared by every sample. Models
            # consume them without B-fold stacking/copying.
            dynamic.update({key: value for key, value in self.topology.items() if key != "bus_type"})
            return dynamic

        n_bus = int(self.topology["bus_type"].numel())
        out = {
            "bus_type": self.topology["bus_type"].repeat(batch_size).unsqueeze(0),
            "Lines_connected": self.topology["Lines_connected"].repeat(batch_size).unsqueeze(0),
            "Y_Lines": self.topology["Y_Lines"].repeat(batch_size).unsqueeze(0),
            "Y_C_Lines": self.topology["Y_C_Lines"].repeat(batch_size).unsqueeze(0),
            "Ybus": torch.block_diag(*([self.topology["Ybus"]] * batch_size)).unsqueeze(0),
            "S_start": dynamic["S_start"].reshape(1, batch_size * n_bus),
            "V_start": dynamic["V_start"].reshape(1, batch_size * n_bus, 2),
            "V_newton": dynamic["V_newton"].reshape(1, batch_size * n_bus, 2),
            "sizes": torch.full((batch_size,), n_bus, dtype=torch.long),
        }
        return out


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


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def forward_pignn_attn_ls_model(model, batch: Dict[str, torch.Tensor], pinn: bool = True):
    sizes = batch.get("sizes")
    out = model(
        batch["bus_type"], batch["Lines_connected"], batch.get("Ybus"),
        batch["Y_Lines"], batch["Y_C_Lines"], batch["S_start"],
        batch["V_start"], sizes,
    )
    return out if pinn else (out, None)


def voltage_supervised_loss(v_pred: torch.Tensor, v_true: torch.Tensor) -> torch.Tensor:
    dvm = v_pred[..., 0] - v_true[..., 0]
    dth = torch.atan2(torch.sin(v_pred[..., 1] - v_true[..., 1]), torch.cos(v_pred[..., 1] - v_true[..., 1]))
    return torch.mean(dvm.square() + dth.square())


def _mismatch_from_voltage(batch: Dict[str, torch.Tensor], v_pred: torch.Tensor):
    vm, va = v_pred[..., 0], v_pred[..., 1]
    vc = vm * torch.exp(1j * va)
    ic = torch.matmul(batch["Ybus"], vc.unsqueeze(-1)).squeeze(-1)
    sc = vc * ic.conj()
    dp = batch["S_start"].real - sc.real
    dq = batch["S_start"].imag - sc.imag
    non_slack = batch["bus_type"] != 1
    pq_mask = batch["bus_type"] == 3
    return dp.masked_fill(~non_slack, 0.0), dq.masked_fill(~pq_mask, 0.0), non_slack, pq_mask


def _sample_metrics(batch, v_pred):
    dp, dq, non_slack, pq_mask = _mismatch_from_voltage(batch, v_pred)
    p = dp.masked_select(non_slack).reshape(v_pred.shape[0], -1)
    q = dq.masked_select(pq_mask).reshape(v_pred.shape[0], -1)
    max_p = p.abs().amax(dim=1)
    max_q = q.abs().amax(dim=1)
    return torch.maximum(max_p, max_q), max_p, max_q, p, q


def _clean_zero(x: float) -> float:
    return 0.0 if abs(x) < 5e-8 else x


def _cell_key(level_row: torch.Tensor) -> Tuple[float, float, float, float]:
    return tuple(_clean_zero(float(v)) for v in level_row.detach().cpu().tolist()[:4])


def _empty_cell_stats():
    return {
        "count": 0, "ill": 0, "ang_max": 0.0, "vol_max": 0.0,
        "p_max": 0.0, "q_max": 0.0, "p_sq_sum": 0.0, "q_sq_sum": 0.0,
        "p_count": 0, "q_count": 0, "mismatch_values": [],
    }


@torch.inference_mode()
def evaluate_pignn_attn_ls_heatmap_cells(model, loader, device, *, model_steps: float, pinn: bool = True):
    del model_steps  # kept for command-line/API compatibility
    model.eval()
    cells = {}
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        v_pred = forward_pignn_attn_ls_model(model, batch, pinn=pinn)[0]
        v_true = batch["V_newton"]
        mismatch, max_p, max_q, p, q = _sample_metrics(batch, v_pred)
        non_slack, pq_mask = batch["bus_type"] != 1, batch["bus_type"] == 3
        dtheta = torch.atan2(torch.sin(v_pred[..., 1] - v_true[..., 1]), torch.cos(v_pred[..., 1] - v_true[..., 1]))
        dvol = v_pred[..., 0] - v_true[..., 0]
        angle_max = dtheta.masked_fill(~non_slack, 0.0).abs().amax(dim=1)
        voltage_max = dvol.masked_fill(~pq_mask, 0.0).abs().amax(dim=1)
        finite = torch.isfinite(v_pred).all(dim=(1, 2))
        for i in range(v_pred.shape[0]):
            stats = cells.setdefault(_cell_key(batch["level"][i]), _empty_cell_stats())
            stats["count"] += 1
            stats["ill"] += int(int(batch["valid_label"][i]) == 0 or not bool(finite[i]))
            stats["ang_max"] = max(stats["ang_max"], float(angle_max[i]))
            stats["vol_max"] = max(stats["vol_max"], float(voltage_max[i]))
            stats["p_max"] = max(stats["p_max"], float(max_p[i]))
            stats["q_max"] = max(stats["q_max"], float(max_q[i]))
            stats["p_sq_sum"] += float(p[i].square().sum())
            stats["q_sq_sum"] += float(q[i].square().sum())
            stats["p_count"] += int(p.shape[1])
            stats["q_count"] += int(q.shape[1])
            stats["mismatch_values"].append(float(mismatch[i]))
    return cells


def _polar_jacobian(ybus, vm, va, bus_type):
    device, dtype = vm.device, vm.dtype
    non = torch.where(bus_type[0] != 1)[0]
    pq = torch.where(bus_type[0] == 3)[0]
    vc = vm * torch.exp(1j * va)
    sc = vc * torch.matmul(ybus, vc.unsqueeze(-1)).squeeze(-1).conj()
    p_calc, q_calc = sc.real, sc.imag
    g, b = ybus.real.to(dtype), ybus.imag.to(dtype)
    n_non, n_pq = int(non.numel()), int(pq.numel())
    jac = torch.empty(vm.shape[0], n_non + n_pq, n_non + n_pq, device=device, dtype=dtype)
    vm_non, va_non = vm[:, non], va[:, non]
    th_nn = va_non.unsqueeze(2) - va_non.unsqueeze(1)
    g_nn, b_nn = g[non][:, non], b[non][:, non]
    h = vm_non.unsqueeze(2) * vm_non.unsqueeze(1) * (g_nn * torch.sin(th_nn) - b_nn * torch.cos(th_nn))
    diag_non = torch.arange(n_non, device=device)
    h[:, diag_non, diag_non] = -q_calc[:, non] - torch.diagonal(b_nn) * vm_non.square()
    jac[:, :n_non, :n_non] = h
    if n_pq:
        vm_pq, va_pq = vm[:, pq], va[:, pq]
        rows = torch.searchsorted(non, pq)
        diag_pq = torch.arange(n_pq, device=device)
        th_np = va_non.unsqueeze(2) - va_pq.unsqueeze(1)
        g_np, b_np = g[non][:, pq], b[non][:, pq]
        n_block = vm_non.unsqueeze(2) * (g_np * torch.cos(th_np) + b_np * torch.sin(th_np))
        n_block[:, rows, diag_pq] = p_calc[:, pq] / vm_pq.clamp_min(1e-8) + torch.diagonal(g[pq][:, pq]) * vm_pq
        jac[:, :n_non, n_non:] = n_block
        th_pn = va_pq.unsqueeze(2) - va_non.unsqueeze(1)
        g_pn, b_pn = g[pq][:, non], b[pq][:, non]
        m_block = -vm_pq.unsqueeze(2) * vm_non.unsqueeze(1) * (g_pn * torch.cos(th_pn) + b_pn * torch.sin(th_pn))
        m_block[:, diag_pq, rows] = p_calc[:, pq] - torch.diagonal(g[pq][:, pq]) * vm_pq.square()
        jac[:, n_non:, :n_non] = m_block
        th_pp = va_pq.unsqueeze(2) - va_pq.unsqueeze(1)
        g_pp, b_pp = g[pq][:, pq], b[pq][:, pq]
        l_block = vm_pq.unsqueeze(2) * (g_pp * torch.sin(th_pp) - b_pp * torch.cos(th_pp))
        l_block[:, diag_pq, diag_pq] = q_calc[:, pq] / vm_pq.clamp_min(1e-8) - torch.diagonal(b_pp) * vm_pq
        jac[:, n_non:, n_non:] = l_block
    return jac, non, pq


@torch.no_grad()
def _nr_refine_until(v_pred, batch, max_steps=20, tol=1e-6):
    vm, va = v_pred[..., 0].clone(), v_pred[..., 1].clone()
    bus_type, ybus = batch["bus_type"], batch["Ybus"]
    slack, pv = bus_type == 1, bus_type == 2
    vm[slack] = batch["V_start"][..., 0][slack]
    va[slack] = batch["V_start"][..., 1][slack]
    vm[pv] = batch["V_start"][..., 0][pv]
    success = torch.zeros(vm.shape[0], dtype=torch.bool, device=vm.device)
    iterations = torch.zeros(vm.shape[0], dtype=torch.long, device=vm.device)
    active = torch.isfinite(vm).all(1) & torch.isfinite(va).all(1)
    for step in range(max_steps + 1):
        pred = torch.stack((vm, va), dim=-1)
        mismatch, _, _, _, _ = _sample_metrics(batch, pred)
        newly = active & (mismatch < tol)
        success |= newly
        iterations[newly & (iterations == 0)] = step
        active &= ~success
        if step == max_steps or not bool(active.any()):
            break
        dp, dq, _, _ = _mismatch_from_voltage(batch, pred)
        jac, non, pq = _polar_jacobian(ybus, vm, va, bus_type)
        rhs = torch.cat((dp[:, non], dq[:, pq]), dim=1)
        sol, info = torch.linalg.solve_ex(jac, rhs.unsqueeze(-1))
        delta = sol.squeeze(-1)
        ok = active & (info == 0) & torch.isfinite(delta).all(1)
        idx = torch.where(ok)[0]
        va[idx[:, None], non] += delta[idx, :non.numel()]
        if pq.numel():
            vm[idx[:, None], pq] = torch.clamp(vm[idx[:, None], pq] + delta[idx, non.numel():], 0.2, 2.5)
        active &= ok
    final = torch.stack((vm, va), dim=-1)
    mismatch, _, _, _, _ = _sample_metrics(batch, final)
    return mismatch, success, iterations


@torch.inference_mode()
def evaluate_pignn_attn_ls_loader(
    model, loader, device, tol: float = 1e-3, pinn: bool = True,
    nr_refine_max_steps: int = 0, nr_refine_tol: float = 1e-6,
    mismatch_csv_path=None,
):
    model.eval()
    max_ps, max_qs, mismatches, sample_ids = [], [], [], []
    nr_mismatches, nr_successes, nr_iters = [], [], []
    total_p_sq = total_q_sq = 0.0
    total_p_count = total_q_count = 0
    voltage_sq = 0.0
    voltage_count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        v_pred = forward_pignn_attn_ls_model(model, batch, pinn=pinn)[0]
        mismatch, max_p, max_q, p, q = _sample_metrics(batch, v_pred)
        max_ps.extend(max_p.cpu().tolist()); max_qs.extend(max_q.cpu().tolist())
        mismatches.extend(mismatch.cpu().tolist()); sample_ids.extend(batch["sample_id"].cpu().tolist())
        total_p_sq += float(p.square().sum()); total_q_sq += float(q.square().sum())
        total_p_count += p.numel(); total_q_count += q.numel()
        finite_label = torch.isfinite(batch["V_newton"]).all(dim=(1, 2))
        if bool(finite_label.any()):
            err = voltage_supervised_loss(v_pred[finite_label], batch["V_newton"][finite_label])
            nodes = int(v_pred[finite_label].shape[0] * v_pred.shape[1])
            voltage_sq += float(err) * nodes; voltage_count += nodes
        if nr_refine_max_steps > 0:
            nr_m, nr_ok, nr_it = _nr_refine_until(v_pred, batch, nr_refine_max_steps, nr_refine_tol)
            nr_mismatches.extend(nr_m.cpu().tolist()); nr_successes.extend(nr_ok.float().cpu().tolist())
            nr_iters.extend(nr_it[nr_ok].cpu().tolist())
    arr = np.asarray(mismatches, dtype=np.float64)
    result = {
        "samples": float(len(arr)),
        "mean_sample_max_abs_dp": float(np.mean(max_ps)), "worst_sample_max_abs_dp": float(np.max(max_ps)),
        "mean_sample_max_abs_dq": float(np.mean(max_qs)), "worst_sample_max_abs_dq": float(np.max(max_qs)),
        "rmse_abs_dp": float(np.sqrt(total_p_sq / max(total_p_count, 1))),
        "rmse_abs_dq": float(np.sqrt(total_q_sq / max(total_q_count, 1))),
        "conv_rate": float(np.mean(arr < tol)), "max_mismatch": float(np.max(arr)),
        "min_mismatch": float(np.min(arr)), "avg_mismatch": float(np.mean(arr)),
        "success_rate_1e_1": float(np.mean(arr < 1e-1)), "success_rate_1e_2": float(np.mean(arr < 1e-2)),
        "success_rate_1e_3": float(np.mean(arr < 1e-3)),
        "voltage_rmse": float(np.sqrt(voltage_sq / max(voltage_count, 1))),
        "nr_refined_avg_mismatch": None, "nr_refined_success_rate_1e_3": None, "nr_refined_avg_iter_1e_3": None,
    }
    if nr_successes:
        ok = np.asarray(nr_successes, dtype=bool)
        nr_arr = np.asarray(nr_mismatches, dtype=np.float64)
        result["nr_refined_avg_mismatch"] = float(nr_arr[ok].mean()) if ok.any() else None
        result["nr_refined_success_rate_1e_3"] = float(ok.mean())
        result["nr_refined_avg_iter_1e_3"] = float(np.mean(nr_iters)) if nr_iters else None
    if mismatch_csv_path is not None:
        write_ill_mismatch_csv(sample_ids, mismatches, mismatch_csv_path)
    return result


def print_metric_summary(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n===== {title} =====")
    print(f"samples                         : {int(metrics['samples'])}")
    print(f"mean max |dP| over non-slack    : {metrics['mean_sample_max_abs_dp']:.6e}")
    print(f"worst max |dP| over non-slack   : {metrics['worst_sample_max_abs_dp']:.6e}")
    print(f"mean max |dQ| over PQ           : {metrics['mean_sample_max_abs_dq']:.6e}")
    print(f"worst max |dQ| over PQ          : {metrics['worst_sample_max_abs_dq']:.6e}")
    print(f"RMSE |dP| over non-slack        : {metrics['rmse_abs_dp']:.6e}")
    print(f"RMSE |dQ| over PQ               : {metrics['rmse_abs_dq']:.6e}")
    print(f"voltage/angle joint RMSE        : {metrics['voltage_rmse']:.6e}")
    print(f"convergence rate                : {metrics['conv_rate']:.6f}")
    print(f"Success rate (<0.1)             : {metrics['success_rate_1e_1']:.6f}")
    print(f"Success rate (<0.01)            : {metrics['success_rate_1e_2']:.6f}")
    print(f"Success rate (<0.001)           : {metrics['success_rate_1e_3']:.6f}")


def print_ill_conditioned_summary(metrics: Optional[Dict[str, float]]) -> None:
    print("\n===== Ill-Conditioned Test Summary =====")
    print("Samples & Maximal mismatch & Minimal mismatch & Avg. mismatch & Success rate (<0.1) & Success rate (<0.01) & Avg. mismatch With N-R PF refine & Success rate With N-R PF refine & Average N-R PF Iteration time")
    if not metrics or not int(metrics.get("samples", 0)):
        print("0 & N/A & N/A & N/A & N/A & N/A & N/A & N/A & N/A")
        return
    def fmt(key):
        value = metrics.get(key)
        return "N/A" if value is None else f"{value:.6e}"
    print(
        f"{int(metrics['samples'])} & {fmt('max_mismatch')} & {fmt('min_mismatch')} & {fmt('avg_mismatch')} & "
        f"{metrics['success_rate_1e_1']:.6f} & {metrics['success_rate_1e_2']:.6f} & "
        f"{fmt('nr_refined_avg_mismatch')} & {fmt('nr_refined_success_rate_1e_3')} & {fmt('nr_refined_avg_iter_1e_3')}"
    )


def print_heatmap_cell_summary(cells) -> None:
    print("\n===== Batch Full-Test Summary By Heatmap Cell =====")
    for key in sorted(cells):
        stats = cells[key]
        p_rmse = np.sqrt(stats["p_sq_sum"] / max(stats["p_count"], 1))
        q_rmse = np.sqrt(stats["q_sq_sum"] / max(stats["q_count"], 1))
        arr = np.asarray(stats["mismatch_values"])
        print(
            f"X=[{key[0]:.1f},{key[1]:.1f}], PQ=[{key[2]:.1f},{key[3]:.1f}]: "
            f"count:{stats['count']}, ill:{stats['ill']}, Ang_MAX:{stats['ang_max']:.6e}, "
            f"Vol_MAX:{stats['vol_max']:.6e}, P_MAX:{stats['p_max']:.3e}, Q_MAX:{stats['q_max']:.3e}, "
            f"P_RMSE:{p_rmse:.3e}, Q_RMSE:{q_rmse:.3e}, "
            f"Success rate (<0.1):{np.mean(arr < 1e-1):.6f}, "
            f"Success rate (<0.01):{np.mean(arr < 1e-2):.6f}, Success rate (<0.001):{np.mean(arr < 1e-3):.6f}"
        )


def print_region_mismatch_summary(cells) -> None:
    regions = {"ID": [], "OOD-X": [], "OOD-PQ": [], "OOD-X+PQ": []}
    for (xl, xh, pl, ph), stats in cells.items():
        xid, pid = xl >= -1.0 - 1e-9 and xh <= 1.0 + 1e-9, pl >= -1.0 - 1e-9 and ph <= 1.0 + 1e-9
        region = "ID" if xid and pid else "OOD-X" if not xid and pid else "OOD-PQ" if xid else "OOD-X+PQ"
        regions[region].extend(stats["mismatch_values"])
    print("\n===== Region Mismatch Summary =====")
    print("Region & Samples & Avg. mismatch & Worst mismatch & Success rate (<0.1) & Success rate (<0.01) & Success rate (<0.001)")
    for name, values in regions.items():
        if not values:
            print(f"{name} & 0 & N/A & N/A & N/A & N/A & N/A")
            continue
        arr = np.asarray(values)
        print(f"{name} & {len(arr)} & {arr.mean():.6e} & {arr.max():.6e} & {np.mean(arr < 1e-1):.6f} & {np.mean(arr < 1e-2):.6f} & {np.mean(arr < 1e-3):.6f}")


def write_heatmap_cell_csv(cells, output_csv) -> None:
    output_csv = Path(output_csv); output_csv.parent.mkdir(parents=True, exist_ok=True)
    xs = sorted({round((k[0] + k[1]) / 2, 10) for k in cells})
    ys = sorted({round((k[2] + k[3]) / 2, 10) for k in cells})
    values = {(round((k[0] + k[1]) / 2, 10), round((k[2] + k[3]) / 2, 10)): max(v["p_max"], v["q_max"]) for k, v in cells.items()}
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["PQ\\X"] + [f"{x:.1f}" for x in xs])
        for y in ys: writer.writerow([f"{y:.1f}"] + [values.get((x, y), "") for x in xs])
    print(f"Heatmap CSV saved: {output_csv.name}")


def write_ill_mismatch_csv(sample_ids, mismatch_values, output_csv) -> None:
    del sample_ids  # SOTA4-compatible CSV schema uses a one-based sample index.
    output_csv = Path(output_csv); output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f); writer.writerow(["sample_index", "mismatch"])
        for idx, mismatch in enumerate(mismatch_values, 1):
            writer.writerow([idx, f"{float(mismatch):.12e}"])
    print(f"Ill-conditioned mismatch CSV saved: {output_csv.name}")
