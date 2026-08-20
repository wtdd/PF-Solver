import csv
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from data import cell_key, idx, mismatch_from_voltage, move_batch, pack_state, polar_jacobian, unpack_state, write_heatmap_csv


def GIN_loss(model, batch, info, ybus):
    pred_state = model(batch, info, ybus)
    target_state = pack_state(batch["vm_true"], batch["va_true"], info)
    # GIN Eq. (18)-(19): MAE supervised training objective.
    return F.l1_loss(pred_state, target_state)


@torch.no_grad()
def predict_voltage(model, batch, info, ybus, device):
    state = model(batch, info, ybus)
    return unpack_state(state, batch, info)


def nr_pf_refine(vm, va, batch, info, ybus, steps: int = 5):
    if steps <= 0:
        return vm, va

    device = vm.device
    non = idx(info.non_slack_idx, device)
    pq = idx(info.pq_idx, device)
    n_non = len(info.non_slack_idx)
    n_pq = len(info.pq_idx)

    vm_ref = vm.clone()
    va_ref = va.clone()
    vm_ref[:, info.slack_idx] = batch["vm_start"][:, info.slack_idx]
    va_ref[:, info.slack_idx] = batch["va_start"][:, info.slack_idx]
    if len(info.pv_idx) > 0:
        pv = idx(info.pv_idx, device)
        vm_ref[:, pv] = batch["vm_start"][:, pv]

    active = torch.isfinite(vm_ref).all(dim=1) & torch.isfinite(va_ref).all(dim=1)
    for _ in range(steps):
        if not bool(active.any().item()):
            break
        dp, dq = mismatch_from_voltage(ybus, batch["p_spec"], batch["q_spec"], vm_ref, va_ref)
        rhs = torch.cat([dp[:, non], dq[:, pq]], dim=1)
        jac = polar_jacobian(ybus, vm_ref, va_ref, info)
        sol, solve_info = torch.linalg.solve_ex(jac, rhs.unsqueeze(-1))
        delta = sol.squeeze(-1)
        ok = active & (solve_info == 0) & torch.isfinite(delta).all(dim=1)
        if not bool(ok.any().item()):
            break

        ok_idx = torch.where(ok)[0]
        va_sub = va_ref[ok_idx].clone()
        va_sub[:, non] = va_sub[:, non] + delta[ok_idx, :n_non]
        va_ref[ok_idx] = va_sub
        if n_pq:
            vm_sub = vm_ref[ok_idx].clone()
            vm_sub[:, pq] = torch.clamp(vm_sub[:, pq] + delta[ok_idx, n_non:], min=0.2, max=2.5)
            vm_ref[ok_idx] = vm_sub
        vm_ref[:, info.slack_idx] = batch["vm_start"][:, info.slack_idx]
        va_ref[:, info.slack_idx] = batch["va_start"][:, info.slack_idx]
        if len(info.pv_idx) > 0:
            vm_ref[:, pv] = batch["vm_start"][:, pv]
        active = active & ok

    return vm_ref, va_ref


def nr_pf_refine_until(vm, va, batch, info, ybus, max_steps: int = 20, tol: float = 1e-3):
    device = vm.device
    non = idx(info.non_slack_idx, device)
    pq = idx(info.pq_idx, device)
    n_non = len(info.non_slack_idx)
    n_pq = len(info.pq_idx)

    vm_ref = vm.clone()
    va_ref = va.clone()
    vm_ref[:, info.slack_idx] = batch["vm_start"][:, info.slack_idx]
    va_ref[:, info.slack_idx] = batch["va_start"][:, info.slack_idx]
    if len(info.pv_idx) > 0:
        pv = idx(info.pv_idx, device)
        vm_ref[:, pv] = batch["vm_start"][:, pv]

    finite = torch.isfinite(vm_ref).all(dim=1) & torch.isfinite(va_ref).all(dim=1)
    init_mismatch, _, _, _, _ = _max_mismatch_values(ybus, batch, vm_ref, va_ref, info, device)
    success = finite & (init_mismatch < tol)
    iterations = torch.zeros(vm_ref.shape[0], dtype=torch.long, device=device)
    active = finite & (~success)

    for step in range(1, max_steps + 1):
        if not bool(active.any().item()):
            break
        dp, dq = mismatch_from_voltage(ybus, batch["p_spec"], batch["q_spec"], vm_ref, va_ref)
        rhs = torch.cat([dp[:, non], dq[:, pq]], dim=1)
        jac = polar_jacobian(ybus, vm_ref, va_ref, info)
        sol, solve_info = torch.linalg.solve_ex(jac, rhs.unsqueeze(-1))
        delta = sol.squeeze(-1)
        ok = active & (solve_info == 0) & torch.isfinite(delta).all(dim=1)
        if not bool(ok.any().item()):
            break

        ok_idx = torch.where(ok)[0]
        va_sub = va_ref[ok_idx].clone()
        va_sub[:, non] = va_sub[:, non] + delta[ok_idx, :n_non]
        va_ref[ok_idx] = va_sub
        if n_pq:
            vm_sub = vm_ref[ok_idx].clone()
            vm_sub[:, pq] = torch.clamp(vm_sub[:, pq] + delta[ok_idx, n_non:], min=0.2, max=2.5)
            vm_ref[ok_idx] = vm_sub
        vm_ref[:, info.slack_idx] = batch["vm_start"][:, info.slack_idx]
        va_ref[:, info.slack_idx] = batch["va_start"][:, info.slack_idx]
        if len(info.pv_idx) > 0:
            vm_ref[:, pv] = batch["vm_start"][:, pv]

        mismatch, _, _, _, _ = _max_mismatch_values(ybus, batch, vm_ref, va_ref, info, device)
        newly_success = ok & (mismatch < tol)
        iterations[newly_success] = step
        success = success | newly_success
        active = ok & (~success)

    return vm_ref, va_ref, success, iterations


def _max_mismatch_values(ybus, batch, vm, va, info, device):
    non = idx(info.non_slack_idx, device)
    pq = idx(info.pq_idx, device)
    dp, dq = mismatch_from_voltage(ybus, batch["p_spec"], batch["q_spec"], vm, va)
    p_batch = dp[:, non]
    q_batch = dq[:, pq]
    max_p = p_batch.abs().amax(dim=1) if p_batch.numel() > 0 else torch.zeros(vm.shape[0], device=device)
    max_q = q_batch.abs().amax(dim=1) if q_batch.numel() > 0 else torch.zeros(vm.shape[0], device=device)
    return torch.maximum(max_p, max_q), max_p, max_q, p_batch, q_batch


def _empty_cell_stats() -> Dict[str, float]:
    return {
        "count": 0,
        "ill": 0,
        "ang_max": 0.0,
        "vol_max": 0.0,
        "p_max": 0.0,
        "q_max": 0.0,
        "ang_sample_sum": 0.0,
        "ang_sample_sq_sum": 0.0,
        "ang_sample_count": 0,
        "vol_sample_sum": 0.0,
        "vol_sample_sq_sum": 0.0,
        "vol_sample_count": 0,
        "p_sample_sum": 0.0,
        "p_sample_sq_sum": 0.0,
        "p_sample_count": 0,
        "q_sample_sum": 0.0,
        "q_sample_sq_sum": 0.0,
        "q_sample_count": 0,
        "mismatch_values": [],
        "p_sq_sum": 0.0,
        "q_sq_sum": 0.0,
        "p_count": 0,
        "q_count": 0,
    }


@torch.no_grad()
def evaluate_full_summary(model, loader, info, ybus, device, tol, nr_refine_max_steps: int = 0, nr_refine_tol: float = 1e-3, mismatch_csv_path=None):
    model.eval()
    max_p_values = []
    max_q_values = []
    max_mismatch_values = []
    nr_refined_mismatch_values = []
    nr_refined_success_values = []
    nr_refined_success_iters = []
    conv_values = []
    total_p_sq = total_q_sq = 0.0
    total_p_count = total_q_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        vm, va = predict_voltage(model, batch, info, ybus, device)
        max_mismatch, max_p, max_q, p_batch, q_batch = _max_mismatch_values(ybus, batch, vm, va, info, device)

        if p_batch.numel() > 0:
            total_p_sq += float(torch.sum(p_batch.square()).item())
            total_p_count += int(p_batch.numel())
        if q_batch.numel() > 0:
            total_q_sq += float(torch.sum(q_batch.square()).item())
            total_q_count += int(q_batch.numel())

        max_p_values.extend(max_p.detach().cpu().tolist())
        max_q_values.extend(max_q.detach().cpu().tolist())
        max_mismatch_values.extend(max_mismatch.detach().cpu().tolist())
        conv_values.extend((max_mismatch < tol).to(torch.float32).detach().cpu().tolist())

        if nr_refine_max_steps > 0:
            vm_nr, va_nr, nr_success, nr_iters = nr_pf_refine_until(vm, va, batch, info, ybus, max_steps=nr_refine_max_steps, tol=nr_refine_tol)
            nr_mismatch, _, _, _, _ = _max_mismatch_values(ybus, batch, vm_nr, va_nr, info, device)
            nr_refined_mismatch_values.extend(nr_mismatch.detach().cpu().tolist())
            nr_refined_success_values.extend(nr_success.to(torch.float32).detach().cpu().tolist())
            if bool(nr_success.any().item()):
                nr_refined_success_iters.extend(nr_iters[nr_success].detach().cpu().tolist())

    if mismatch_csv_path is not None:
        write_ill_mismatch_csv(max_mismatch_values, mismatch_csv_path)

    nr_mismatch_arr = np.asarray(nr_refined_mismatch_values, dtype=np.float64)
    nr_success_arr = np.asarray(nr_refined_success_values, dtype=np.float64)
    nr_iter_arr = np.asarray(nr_refined_success_iters, dtype=np.float64)
    nr_success_mask = nr_success_arr.astype(bool) if nr_success_arr.size else np.asarray([], dtype=bool)
    mismatch_arr = np.asarray(max_mismatch_values, dtype=np.float64)
    return {
        "samples": float(len(max_p_values)),
        "mean_sample_max_abs_dp": float(np.mean(max_p_values)),
        "worst_sample_max_abs_dp": float(max(max_p_values)),
        "mean_sample_max_abs_dq": float(np.mean(max_q_values)),
        "worst_sample_max_abs_dq": float(max(max_q_values)),
        "rmse_abs_dp": float(np.sqrt(total_p_sq / max(total_p_count, 1))),
        "rmse_abs_dq": float(np.sqrt(total_q_sq / max(total_q_count, 1))),
        "conv_rate": float(np.mean(conv_values)),
        "max_mismatch": float(np.max(mismatch_arr)),
        "min_mismatch": float(np.min(mismatch_arr)),
        "avg_mismatch": float(np.mean(mismatch_arr)),
        "success_rate_1e_1": float(np.mean(mismatch_arr < 1e-1)),
        "success_rate_1e_2": float(np.mean(mismatch_arr < 1e-2)),
        "success_rate_1e_3": float(np.mean(mismatch_arr < 1e-3)),
        "nr_refined_avg_mismatch": float(np.mean(nr_mismatch_arr[nr_success_mask])) if nr_success_mask.any() else None,
        "nr_refined_success_rate_1e_3": float(np.mean(nr_success_arr)) if nr_success_arr.size else None,
        "nr_refined_avg_iter_1e_3": float(np.mean(nr_iter_arr)) if nr_iter_arr.size else None,
    }


@torch.no_grad()
def evaluate_heatmap_cells(model, loader, info, ybus, device):
    model.eval()
    non = idx(info.non_slack_idx, device)
    pq = idx(info.pq_idx, device)
    cells = {}
    for batch in loader:
        batch = move_batch(batch, device)
        vm, va = predict_voltage(model, batch, info, ybus, device)
        dp, dq = mismatch_from_voltage(ybus, batch["p_spec"], batch["q_spec"], vm, va)
        dtheta = torch.atan2(torch.sin(va - batch["va_true"]), torch.cos(va - batch["va_true"]))
        dvol = vm - batch["vm_true"]

        angle_batch = dtheta[:, non].abs()
        voltage_batch = dvol[:, pq].abs()
        p_batch = dp[:, non]
        q_batch = dq[:, pq]
        bsz = vm.shape[0]

        angle_max = angle_batch.amax(dim=1) if angle_batch.numel() > 0 else torch.zeros(bsz, device=device)
        voltage_max = voltage_batch.amax(dim=1) if voltage_batch.numel() > 0 else torch.zeros(bsz, device=device)
        p_abs_max = p_batch.abs().amax(dim=1) if p_batch.numel() > 0 else torch.zeros(bsz, device=device)
        q_abs_max = q_batch.abs().amax(dim=1) if q_batch.numel() > 0 else torch.zeros(bsz, device=device)
        p_sq_sum = p_batch.square().sum(dim=1) if p_batch.numel() > 0 else torch.zeros(bsz, device=device)
        q_sq_sum = q_batch.square().sum(dim=1) if q_batch.numel() > 0 else torch.zeros(bsz, device=device)
        finite_state = torch.isfinite(vm).all(dim=1) & torch.isfinite(va).all(dim=1)

        levels = batch["level"].detach().cpu()
        valid_labels = batch["valid_label"].detach().cpu().numpy()
        finite_state_np = finite_state.detach().cpu().numpy()
        angle_np = angle_max.detach().cpu().numpy()
        voltage_np = voltage_max.detach().cpu().numpy()
        p_max_np = p_abs_max.detach().cpu().numpy()
        q_max_np = q_abs_max.detach().cpu().numpy()
        p_sq_np = p_sq_sum.detach().cpu().numpy()
        q_sq_np = q_sq_sum.detach().cpu().numpy()
        p_count = int(p_batch.shape[1]) if p_batch.ndim == 2 else 0
        q_count = int(q_batch.shape[1]) if q_batch.ndim == 2 else 0

        for i in range(bsz):
            key = cell_key(levels[i])
            stats = cells.setdefault(key, _empty_cell_stats())
            is_ill = (int(valid_labels[i]) == 0) or (not bool(finite_state_np[i]))

            sample_ang = float(angle_np[i])
            sample_vol = float(voltage_np[i])
            sample_p = float(p_max_np[i])
            sample_q = float(q_max_np[i])

            stats["count"] += 1
            stats["ill"] += int(is_ill)
            stats["ang_max"] = max(stats["ang_max"], sample_ang)
            stats["vol_max"] = max(stats["vol_max"], sample_vol)
            stats["p_max"] = max(stats["p_max"], sample_p)
            stats["q_max"] = max(stats["q_max"], sample_q)
            stats["ang_sample_sum"] += sample_ang
            stats["ang_sample_sq_sum"] += sample_ang * sample_ang
            stats["ang_sample_count"] += 1
            stats["vol_sample_sum"] += sample_vol
            stats["vol_sample_sq_sum"] += sample_vol * sample_vol
            stats["vol_sample_count"] += 1
            stats["p_sample_sum"] += sample_p
            stats["p_sample_sq_sum"] += sample_p * sample_p
            stats["p_sample_count"] += 1
            stats["q_sample_sum"] += sample_q
            stats["q_sample_sq_sum"] += sample_q * sample_q
            stats["q_sample_count"] += 1
            stats["p_sq_sum"] += float(p_sq_np[i])
            stats["q_sq_sum"] += float(q_sq_np[i])
            stats["p_count"] += p_count
            stats["q_count"] += q_count
            stats["mismatch_values"].append(max(sample_p, sample_q))
    return cells


def print_metric_summary(title: str, metrics):
    print(f"\n===== {title} =====")
    print(f"samples                         : {int(metrics['samples'])}")
    print(f"mean max |dP| over non-slack    : {metrics['mean_sample_max_abs_dp']:.6e}")
    print(f"worst max |dP| over non-slack   : {metrics['worst_sample_max_abs_dp']:.6e}")
    print(f"mean max |dQ| over PQ           : {metrics['mean_sample_max_abs_dq']:.6e}")
    print(f"worst max |dQ| over PQ          : {metrics['worst_sample_max_abs_dq']:.6e}")
    print(f"RMSE |dP| over non-slack        : {metrics['rmse_abs_dp']:.6e}")
    print(f"RMSE |dQ| over PQ               : {metrics['rmse_abs_dq']:.6e}")
    print(f"convergence rate                : {metrics['conv_rate']:.6f}")
    print(f"Success rate (<0.1)             : {metrics['success_rate_1e_1']:.6f}")
    print(f"Success rate (<0.01)            : {metrics['success_rate_1e_2']:.6f}")
    print(f"Success rate (<0.001)           : {metrics['success_rate_1e_3']:.6f}")


def print_ill_conditioned_summary(metrics):
    print("\n===== Ill-Conditioned Test Summary =====")
    print("Samples & Maximal mismatch & Minimal mismatch & Avg. mismatch & Success rate (<0.1) & Success rate (<0.01) & Avg. mismatch With N-R PF refine & Success rate With N-R PF refine & Average N-R PF Iteration time")
    if metrics is None or int(metrics.get("samples", 0)) == 0:
        print("0 & N/A & N/A & N/A & N/A & N/A & N/A & N/A & N/A")
        return
    max_mismatch = metrics.get("max_mismatch")
    max_mismatch_str = "N/A" if max_mismatch is None else f"{max_mismatch:.6e}"
    min_mismatch = metrics.get("min_mismatch")
    min_mismatch_str = "N/A" if min_mismatch is None else f"{min_mismatch:.6e}"
    avg_mismatch = metrics.get("avg_mismatch")
    avg_mismatch_str = "N/A" if avg_mismatch is None else f"{avg_mismatch:.6e}"
    nr_avg_mismatch = metrics.get("nr_refined_avg_mismatch")
    nr_avg_mismatch_str = "N/A" if nr_avg_mismatch is None else f"{nr_avg_mismatch:.6e}"
    nr_success = metrics.get("nr_refined_success_rate_1e_3")
    nr_success_str = "N/A" if nr_success is None else f"{nr_success:.6f}"
    nr_avg_iter = metrics.get("nr_refined_avg_iter_1e_3")
    nr_avg_iter_str = "N/A" if nr_avg_iter is None else f"{nr_avg_iter:.6f}"
    print(
        f"{int(metrics['samples'])} & {max_mismatch_str} & {min_mismatch_str} & {avg_mismatch_str} & "
        f"{metrics['success_rate_1e_1']:.6f} & {metrics['success_rate_1e_2']:.6f} & "
        f"{nr_avg_mismatch_str} & {nr_success_str} & {nr_avg_iter_str}"
    )


def print_heatmap_cell_summary(cells):
    def mean_std(stats, prefix):
        count = max(int(stats[f"{prefix}_sample_count"]), 1)
        mean = float(stats[f"{prefix}_sample_sum"] / count)
        mean_sq = float(stats[f"{prefix}_sample_sq_sum"] / count)
        return mean, float(np.sqrt(max(mean_sq - mean * mean, 0.0)))

    print("\n===== Batch Full-Test Summary By Heatmap Cell =====")
    for x_low, x_high, pq_low, pq_high in sorted(cells.keys()):
        stats = cells[(x_low, x_high, pq_low, pq_high)]
        p_rmse = float(np.sqrt(stats["p_sq_sum"] / max(int(stats["p_count"]), 1)))
        q_rmse = float(np.sqrt(stats["q_sq_sum"] / max(int(stats["q_count"]), 1)))
        ang_mean, ang_std = mean_std(stats, "ang")
        vol_mean, vol_std = mean_std(stats, "vol")
        p_mean, p_std = mean_std(stats, "p")
        q_mean, q_std = mean_std(stats, "q")
        mismatch_arr = np.asarray(stats["mismatch_values"], dtype=np.float64)
        success_1e_1 = float(np.mean(mismatch_arr < 1e-1)) if mismatch_arr.size else 0.0
        success_1e_2 = float(np.mean(mismatch_arr < 1e-2)) if mismatch_arr.size else 0.0
        success_1e_3 = float(np.mean(mismatch_arr < 1e-3)) if mismatch_arr.size else 0.0
        print(
            f"X=[{x_low:.1f},{x_high:.1f}], PQ=[{pq_low:.1f},{pq_high:.1f}]: "
            f"count:{int(stats['count'])}, ill:{int(stats['ill'])}, "
            f"Ang_MAX:{stats['ang_max']:.6e}, Vol_MAX:{stats['vol_max']:.6e}, "
            f"Ang_mean±std:{ang_mean:.6e}±{ang_std:.6e}, "
            f"Vol_mean±std:{vol_mean:.6e}±{vol_std:.6e}, "
            f"P_MAX:{stats['p_max']:.3e}, Q_MAX:{stats['q_max']:.3e}, "
            f"P_mean±std:{p_mean:.3e}±{p_std:.3e}, "
            f"Q_mean±std:{q_mean:.3e}±{q_std:.3e}, "
            f"P_RMSE:{p_rmse:.3e}, Q_RMSE:{q_rmse:.3e}, "
            f"Success rate (<0.1):{success_1e_1:.6f}, "
            f"Success rate (<0.01):{success_1e_2:.6f}, "
            f"Success rate (<0.001):{success_1e_3:.6f}"
        )


def _axis_in_train_space(low: float, high: float) -> bool:
    return float(low) >= -1.0 - 1e-9 and float(high) <= 1.0 + 1e-9


def _region_name(x_low: float, x_high: float, pq_low: float, pq_high: float) -> str:
    x_id = _axis_in_train_space(x_low, x_high)
    pq_id = _axis_in_train_space(pq_low, pq_high)
    if x_id and pq_id:
        return "ID"
    if (not x_id) and pq_id:
        return "OOD-X"
    if x_id and (not pq_id):
        return "OOD-PQ"
    return "OOD-X+PQ"


def print_region_mismatch_summary(cells, tol: float = 1e-3):
    regions = {"ID": [], "OOD-X": [], "OOD-PQ": [], "OOD-X+PQ": []}
    for x_low, x_high, pq_low, pq_high in sorted(cells.keys()):
        region = _region_name(x_low, x_high, pq_low, pq_high)
        regions[region].extend(float(v) for v in cells[(x_low, x_high, pq_low, pq_high)]["mismatch_values"])

    print("\n===== Region Mismatch Summary =====")
    print("Region & Samples & Avg. mismatch & Worst mismatch & Success rate (<0.1) & Success rate (<0.01) & Success rate (<0.001) ")
    for region in ("ID", "OOD-X", "OOD-PQ", "OOD-X+PQ"):
        values = regions[region]
        if not values:
            print(f"{region} & 0 & N/A & N/A & N/A & N/A & N/A ")
            continue
        arr = np.asarray(values, dtype=np.float64)
        success_1e_1 = float(np.mean(arr < 1e-1))
        success_1e_2 = float(np.mean(arr < 1e-2))
        success_1e_3 = float(np.mean(arr < 1e-3))
        print(f"{region} & {len(values)} & {float(arr.mean()):.6e} & {float(arr.max()):.6e} & {success_1e_1:.6f} & {success_1e_2:.6f} & {success_1e_3:.6f} \\\\")


def write_heatmap_cell_csv(cells, output_csv):
    write_heatmap_csv(cells, output_csv)


def write_ill_mismatch_csv(mismatch_values, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "mismatch"])
        for sample_idx, mismatch in enumerate(mismatch_values, start=1):
            writer.writerow([sample_idx, f"{float(mismatch):.12e}"])
