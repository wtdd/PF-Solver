import csv
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from data import (
    cell_key,
    idx,
    mismatch_from_voltage,
    move_batch,
    polar_jacobian,
    write_heatmap_csv,
)


def TGN_loss(model, batch, info, ybus):
    """Paper Eq. (21): unsupervised mean squared AC power imbalance."""
    vm, va = model.forward_voltage(batch, ybus)
    dp, dq = mismatch_from_voltage(ybus, batch["p_spec"], batch["q_spec"], vm, va)
    non = idx(info.non_slack_idx, vm.device)
    pq = idx(info.pq_idx, vm.device)
    # Slack P/Q and PV Q are locally compensated, so their residuals are zero.
    residual_sum = dp[:, non].square().sum() + dq[:, pq].square().sum()
    return residual_sum / max(int(vm.shape[0] * info.n_bus), 1)


@torch.no_grad()
def predict_voltage(model, batch, info, ybus, device):
    del info, device
    return model.forward_voltage(batch, ybus)


def _max_mismatch_values(ybus, batch, vm, va, info, device):
    non = idx(info.non_slack_idx, device)
    pq = idx(info.pq_idx, device)
    dp, dq = mismatch_from_voltage(ybus, batch["p_spec"], batch["q_spec"], vm, va)
    p_batch = dp[:, non]
    q_batch = dq[:, pq]
    zeros = torch.zeros(vm.shape[0], device=device)
    max_p = p_batch.abs().amax(dim=1) if p_batch.numel() else zeros
    max_q = q_batch.abs().amax(dim=1) if q_batch.numel() else zeros
    return torch.maximum(max_p, max_q), max_p, max_q, p_batch, q_batch


class _StreamingPairMoments:
    """Numerically stable streaming moments for Pearson correlation."""

    def __init__(self):
        self.count = 0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.m2_x = 0.0
        self.m2_y = 0.0
        self.cross = 0.0

    def update(self, x: torch.Tensor, y: torch.Tensor):
        x = x.reshape(-1).to(torch.float64)
        y = y.reshape(-1).to(torch.float64)
        batch_count = int(x.numel())
        if batch_count == 0:
            return
        batch_mean_x = float(x.mean().item())
        batch_mean_y = float(y.mean().item())
        centered_x = x - batch_mean_x
        centered_y = y - batch_mean_y
        batch_m2_x = float(centered_x.square().sum().item())
        batch_m2_y = float(centered_y.square().sum().item())
        batch_cross = float((centered_x * centered_y).sum().item())

        if self.count == 0:
            self.count = batch_count
            self.mean_x = batch_mean_x
            self.mean_y = batch_mean_y
            self.m2_x = batch_m2_x
            self.m2_y = batch_m2_y
            self.cross = batch_cross
            return

        total_count = self.count + batch_count
        delta_x = batch_mean_x - self.mean_x
        delta_y = batch_mean_y - self.mean_y
        correction = self.count * batch_count / total_count
        self.cross += batch_cross + delta_x * delta_y * correction
        self.m2_x += batch_m2_x + delta_x * delta_x * correction
        self.m2_y += batch_m2_y + delta_y * delta_y * correction
        self.mean_x += delta_x * batch_count / total_count
        self.mean_y += delta_y * batch_count / total_count
        self.count = total_count

    def correlation(self):
        denominator = np.sqrt(max(self.m2_x, 0.0) * max(self.m2_y, 0.0))
        if self.count <= 1 or denominator <= 0.0:
            return None
        return float(self.cross / denominator)


def nr_pf_refine_until(vm, va, batch, info, ybus, max_steps: int = 20, tol: float = 1e-3):
    device = vm.device
    non = idx(info.non_slack_idx, device)
    pq = idx(info.pq_idx, device)
    n_non = len(info.non_slack_idx)
    n_pq = len(info.pq_idx)
    pv = idx(info.pv_idx, device)

    vm_ref = vm.clone()
    va_ref = va.clone()
    vm_ref[:, info.slack_idx] = batch["vm_start"][:, info.slack_idx]
    va_ref[:, info.slack_idx] = batch["va_start"][:, info.slack_idx]
    if pv.numel():
        vm_ref[:, pv] = batch["vm_start"][:, pv]

    finite = torch.isfinite(vm_ref).all(dim=1) & torch.isfinite(va_ref).all(dim=1)
    initial, _, _, _, _ = _max_mismatch_values(ybus, batch, vm_ref, va_ref, info, device)
    success = finite & (initial < tol)
    iterations = torch.zeros(vm_ref.shape[0], dtype=torch.long, device=device)
    active = finite & (~success)

    for step in range(1, max_steps + 1):
        if not bool(active.any().item()):
            break
        dp, dq = mismatch_from_voltage(ybus, batch["p_spec"], batch["q_spec"], vm_ref, va_ref)
        rhs = torch.cat([dp[:, non], dq[:, pq]], dim=1)
        jacobian = polar_jacobian(ybus, vm_ref, va_ref, info)
        solution, solve_info = torch.linalg.solve_ex(jacobian, rhs.unsqueeze(-1))
        delta = solution.squeeze(-1)
        ok = active & (solve_info == 0) & torch.isfinite(delta).all(dim=1)
        if not bool(ok.any().item()):
            break

        ok_rows = torch.where(ok)[0]
        va_sub = va_ref[ok_rows].clone()
        va_sub[:, non] += delta[ok_rows, :n_non]
        va_ref[ok_rows] = va_sub
        if n_pq:
            vm_sub = vm_ref[ok_rows].clone()
            vm_sub[:, pq] = torch.clamp(vm_sub[:, pq] + delta[ok_rows, n_non:], min=0.2, max=2.5)
            vm_ref[ok_rows] = vm_sub
        vm_ref[:, info.slack_idx] = batch["vm_start"][:, info.slack_idx]
        va_ref[:, info.slack_idx] = batch["va_start"][:, info.slack_idx]
        if pv.numel():
            vm_ref[:, pv] = batch["vm_start"][:, pv]

        mismatch, _, _, _, _ = _max_mismatch_values(ybus, batch, vm_ref, va_ref, info, device)
        newly_success = ok & (mismatch < tol)
        iterations[newly_success] = step
        success |= newly_success
        active = ok & (~success)
    return vm_ref, va_ref, success, iterations


@torch.no_grad()
def evaluate_full_summary(
    model,
    loader,
    info,
    ybus,
    device,
    tol,
    nr_refine_max_steps: int = 0,
    nr_refine_tol: float = 1e-3,
    mismatch_csv_path=None,
    compute_voltage_metrics: bool = True,
):
    model.eval()
    max_p_values = []
    max_q_values = []
    mismatch_values = []
    refined_mismatch_values = []
    refined_success_values = []
    refined_iterations = []
    total_p_sq = total_q_sq = 0.0
    total_p_count = total_q_count = 0
    voltage_reference_count = 0
    voltage_sample_count = 0
    voltage_value_count = 0
    vm_error_sq_sum = va_error_sq_sum = 0.0
    vm_moments = _StreamingPairMoments()
    va_moments = _StreamingPairMoments()

    for batch in loader:
        batch = move_batch(batch, device)
        vm, va = predict_voltage(model, batch, info, ybus, device)
        mismatch, max_p, max_q, p_batch, q_batch = _max_mismatch_values(ybus, batch, vm, va, info, device)
        max_p_values.extend(max_p.cpu().tolist())
        max_q_values.extend(max_q.cpu().tolist())
        mismatch_values.extend(mismatch.cpu().tolist())
        total_p_sq += float(p_batch.square().sum().item())
        total_q_sq += float(q_batch.square().sum().item())
        total_p_count += int(p_batch.numel())
        total_q_count += int(q_batch.numel())

        if compute_voltage_metrics:
            reference_valid = (
                torch.isfinite(batch["vm_true"]).all(dim=1)
                & torch.isfinite(batch["va_true"]).all(dim=1)
            )
            if "valid_label" in batch:
                reference_valid &= batch["valid_label"].to(torch.bool)
            comparable = (
                reference_valid
                & torch.isfinite(vm).all(dim=1)
                & torch.isfinite(va).all(dim=1)
            )
            voltage_reference_count += int(reference_valid.sum().item())
            if bool(comparable.any().item()):
                vm_valid = vm[comparable].to(torch.float64)
                va_valid = va[comparable].to(torch.float64)
                vm_true = batch["vm_true"][comparable].to(torch.float64)
                va_true = batch["va_true"][comparable].to(torch.float64)
                voltage_sample_count += int(comparable.sum().item())
                voltage_value_count += int(vm_valid.numel())
                vm_error_sq_sum += float((vm_valid - vm_true).square().sum().item())
                wrapped_va_error = torch.atan2(
                    torch.sin(va_valid - va_true),
                    torch.cos(va_valid - va_true),
                )
                va_aligned = va_true + wrapped_va_error
                va_error_sq_sum += float(wrapped_va_error.square().sum().item())
                vm_moments.update(vm_valid, vm_true)
                va_moments.update(va_aligned, va_true)

        if nr_refine_max_steps > 0:
            vm_nr, va_nr, nr_success, nr_iters = nr_pf_refine_until(
                vm, va, batch, info, ybus, max_steps=nr_refine_max_steps, tol=nr_refine_tol
            )
            nr_mismatch, _, _, _, _ = _max_mismatch_values(ybus, batch, vm_nr, va_nr, info, device)
            refined_mismatch_values.extend(nr_mismatch.cpu().tolist())
            refined_success_values.extend(nr_success.float().cpu().tolist())
            if bool(nr_success.any().item()):
                refined_iterations.extend(nr_iters[nr_success].cpu().tolist())

    if mismatch_csv_path is not None:
        write_ill_mismatch_csv(mismatch_values, mismatch_csv_path)
    if not mismatch_values:
        return None

    mismatch_arr = np.asarray(mismatch_values, dtype=np.float64)
    nr_mismatch = np.asarray(refined_mismatch_values, dtype=np.float64)
    nr_success = np.asarray(refined_success_values, dtype=np.float64)
    nr_iterations = np.asarray(refined_iterations, dtype=np.float64)
    nr_mask = nr_success.astype(bool) if nr_success.size else np.asarray([], dtype=bool)
    return {
        "samples": float(len(mismatch_values)),
        "mean_sample_max_abs_dp": float(np.mean(max_p_values)),
        "worst_sample_max_abs_dp": float(np.max(max_p_values)),
        "mean_sample_max_abs_dq": float(np.mean(max_q_values)),
        "worst_sample_max_abs_dq": float(np.max(max_q_values)),
        "rmse_abs_dp": float(np.sqrt(total_p_sq / max(total_p_count, 1))),
        "rmse_abs_dq": float(np.sqrt(total_q_sq / max(total_q_count, 1))),
        "conv_rate": float(np.mean(mismatch_arr < tol)),
        "max_mismatch": float(np.max(mismatch_arr)),
        "min_mismatch": float(np.min(mismatch_arr)),
        "avg_mismatch": float(np.mean(mismatch_arr)),
        "success_rate_1e_1": float(np.mean(mismatch_arr < 1e-1)),
        "success_rate_1e_2": float(np.mean(mismatch_arr < 1e-2)),
        "success_rate_1e_3": float(np.mean(mismatch_arr < 1e-3)),
        "paper_voltage_reference_samples": float(voltage_reference_count) if compute_voltage_metrics else None,
        "paper_voltage_samples": float(voltage_sample_count) if compute_voltage_metrics else None,
        "paper_voltage_prediction_failures": float(voltage_reference_count - voltage_sample_count) if compute_voltage_metrics else None,
        "paper_voltage_finite_rate": (
            float(voltage_sample_count / voltage_reference_count) if voltage_reference_count else None
        ) if compute_voltage_metrics else None,
        "paper_vm_rmse": float(np.sqrt(vm_error_sq_sum / voltage_value_count)) if voltage_value_count else None,
        "paper_va_rmse": float(np.sqrt(va_error_sq_sum / voltage_value_count)) if voltage_value_count else None,
        "paper_vm_corr": vm_moments.correlation() if voltage_value_count else None,
        "paper_va_corr": va_moments.correlation() if voltage_value_count else None,
        "nr_refined_avg_mismatch": float(np.mean(nr_mismatch[nr_mask])) if nr_mask.any() else None,
        "nr_refined_success_rate_1e_3": float(np.mean(nr_success)) if nr_success.size else None,
        "nr_refined_avg_iter_1e_3": float(np.mean(nr_iterations)) if nr_iterations.size else None,
    }


def _empty_cell_stats() -> Dict[str, object]:
    stats = {"count": 0, "ill": 0, "ang_max": 0.0, "vol_max": 0.0, "p_max": 0.0, "q_max": 0.0}
    for prefix in ("ang", "vol", "p", "q"):
        stats[f"{prefix}_sample_sum"] = 0.0
        stats[f"{prefix}_sample_sq_sum"] = 0.0
        stats[f"{prefix}_sample_count"] = 0
    stats.update({"mismatch_values": [], "p_sq_sum": 0.0, "q_sq_sum": 0.0, "p_count": 0, "q_count": 0})
    return stats


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
        angle = dtheta[:, non].abs()
        voltage = (vm - batch["vm_true"])[:, pq].abs()
        p_batch = dp[:, non]
        q_batch = dq[:, pq]
        batch_size = vm.shape[0]
        zeros = torch.zeros(batch_size, device=device)
        fields = {
            "ang": angle.amax(dim=1) if angle.numel() else zeros,
            "vol": voltage.amax(dim=1) if voltage.numel() else zeros,
            "p": p_batch.abs().amax(dim=1) if p_batch.numel() else zeros,
            "q": q_batch.abs().amax(dim=1) if q_batch.numel() else zeros,
        }
        field_arrays = {name: value.cpu().numpy() for name, value in fields.items()}
        p_sq = p_batch.square().sum(dim=1).cpu().numpy()
        q_sq = q_batch.square().sum(dim=1).cpu().numpy()
        finite = (torch.isfinite(vm).all(dim=1) & torch.isfinite(va).all(dim=1)).cpu().numpy()
        valid = batch["valid_label"].cpu().numpy()
        levels = batch["level"].cpu()

        for row in range(batch_size):
            stats = cells.setdefault(cell_key(levels[row]), _empty_cell_stats())
            values = {name: float(array[row]) for name, array in field_arrays.items()}
            stats["count"] += 1
            stats["ill"] += int(int(valid[row]) == 0 or not bool(finite[row]))
            for name, value in values.items():
                max_name = "vol_max" if name == "vol" else f"{name}_max"
                stats[max_name] = max(float(stats[max_name]), value)
                stats[f"{name}_sample_sum"] += value
                stats[f"{name}_sample_sq_sum"] += value * value
                stats[f"{name}_sample_count"] += 1
            stats["p_sq_sum"] += float(p_sq[row])
            stats["q_sq_sum"] += float(q_sq[row])
            stats["p_count"] += int(p_batch.shape[1])
            stats["q_count"] += int(q_batch.shape[1])
            stats["mismatch_values"].append(max(values["p"], values["q"]))
    return cells


def print_metric_summary(title: str, metrics):
    print(f"\n===== {title} =====")
    if metrics is None:
        print("No samples")
        return
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
    if metrics.get("paper_vm_rmse") is not None:
        print("\n===== Paper Eq. (22) Voltage Accuracy =====")
        print(f"valid N-R reference samples     : {int(metrics['paper_voltage_reference_samples'])}")
        print(f"finite prediction samples       : {int(metrics['paper_voltage_samples'])}")
        print(f"non-finite prediction failures  : {int(metrics['paper_voltage_prediction_failures'])}")
        print(f"prediction finite coverage      : {metrics['paper_voltage_finite_rate']:.6f}")
        print(f"voltage magnitude RMSE (p.u.)   : {metrics['paper_vm_rmse']:.6e}")
        print(f"voltage phase RMSE (rad)        : {metrics['paper_va_rmse']:.6e}")
        vm_corr = metrics.get("paper_vm_corr")
        va_corr = metrics.get("paper_va_corr")
        print(f"voltage magnitude correlation   : {'N/A' if vm_corr is None else f'{vm_corr:.6f}'}")
        print(f"voltage phase correlation       : {'N/A' if va_corr is None else f'{va_corr:.6f}'}")


def print_ill_conditioned_summary(metrics):
    print("\n===== Ill-Conditioned Test Summary =====")
    print("Samples & Maximal mismatch & Minimal mismatch & Avg. mismatch & Success rate (<0.1) & Success rate (<0.01) & Avg. mismatch With N-R PF refine (success-only) & Success rate With N-R PF refine & Average N-R PF Iteration time")
    if not metrics:
        print("0 & N/A & N/A & N/A & N/A & N/A & N/A & N/A & N/A")
        return
    optional = lambda key: "N/A" if metrics.get(key) is None else f"{metrics[key]:.6e}"
    nr_success = "N/A" if metrics.get("nr_refined_success_rate_1e_3") is None else f"{metrics['nr_refined_success_rate_1e_3']:.6f}"
    nr_iter = "N/A" if metrics.get("nr_refined_avg_iter_1e_3") is None else f"{metrics['nr_refined_avg_iter_1e_3']:.6f}"
    print(
        f"{int(metrics['samples'])} & {optional('max_mismatch')} & {optional('min_mismatch')} & "
        f"{optional('avg_mismatch')} & {metrics['success_rate_1e_1']:.6f} & {metrics['success_rate_1e_2']:.6f} & "
        f"{optional('nr_refined_avg_mismatch')} & {nr_success} & {nr_iter}"
    )
    if metrics.get("paper_vm_rmse") is not None:
        vm_corr = metrics.get("paper_vm_corr")
        va_corr = metrics.get("paper_va_corr")
        print("\nRaw TGN voltage accuracy on all comparable ill-conditioned samples (not N-R success-only):")
        print(
            f"reference={int(metrics['paper_voltage_reference_samples'])}, "
            f"evaluated={int(metrics['paper_voltage_samples'])}, "
            f"prediction_failures={int(metrics['paper_voltage_prediction_failures'])}, "
            f"finite_coverage={metrics['paper_voltage_finite_rate']:.6f}"
        )
        print(
            f"Eq.(22) Vm RMSE={metrics['paper_vm_rmse']:.6e} p.u., "
            f"wrapped Va RMSE={metrics['paper_va_rmse']:.6e} rad, "
            f"Vm Pearson r={'N/A' if vm_corr is None else f'{vm_corr:.6f}'}, "
            f"wrapped Va Pearson r={'N/A' if va_corr is None else f'{va_corr:.6f}'}"
        )


def _mean_std(stats, prefix):
    count = max(int(stats[f"{prefix}_sample_count"]), 1)
    mean = float(stats[f"{prefix}_sample_sum"] / count)
    mean_sq = float(stats[f"{prefix}_sample_sq_sum"] / count)
    return mean, float(np.sqrt(max(mean_sq - mean * mean, 0.0)))


def print_heatmap_cell_summary(cells):
    print("\n===== Batch Full-Test Summary By Heatmap Cell =====")
    for key in sorted(cells):
        stats = cells[key]
        x_low, x_high, pq_low, pq_high = key
        means = {name: _mean_std(stats, name) for name in ("ang", "vol", "p", "q")}
        arr = np.asarray(stats["mismatch_values"], dtype=np.float64)
        p_rmse = np.sqrt(stats["p_sq_sum"] / max(int(stats["p_count"]), 1))
        q_rmse = np.sqrt(stats["q_sq_sum"] / max(int(stats["q_count"]), 1))
        print(
            f"X=[{x_low:.1f},{x_high:.1f}], PQ=[{pq_low:.1f},{pq_high:.1f}]: count:{int(stats['count'])}, "
            f"ill:{int(stats['ill'])}, Ang_MAX:{stats['ang_max']:.6e}, Vol_MAX:{stats['vol_max']:.6e}, "
            f"Ang_mean+-std:{means['ang'][0]:.6e}+-{means['ang'][1]:.6e}, Vol_mean+-std:{means['vol'][0]:.6e}+-{means['vol'][1]:.6e}, "
            f"P_MAX:{stats['p_max']:.3e}, Q_MAX:{stats['q_max']:.3e}, P_mean+-std:{means['p'][0]:.3e}+-{means['p'][1]:.3e}, "
            f"Q_mean+-std:{means['q'][0]:.3e}+-{means['q'][1]:.3e}, P_RMSE:{p_rmse:.3e}, Q_RMSE:{q_rmse:.3e}, "
            f"Success rate (<0.1):{np.mean(arr < 1e-1):.6f}, Success rate (<0.01):{np.mean(arr < 1e-2):.6f}, "
            f"Success rate (<0.001):{np.mean(arr < 1e-3):.6f}"
        )


def _axis_in_train_space(low: float, high: float) -> bool:
    return float(low) >= -1.0 - 1e-9 and float(high) <= 1.0 + 1e-9


def _region_name(key) -> str:
    x_id = _axis_in_train_space(key[0], key[1])
    pq_id = _axis_in_train_space(key[2], key[3])
    return "ID" if x_id and pq_id else "OOD-X" if not x_id and pq_id else "OOD-PQ" if x_id else "OOD-X+PQ"


def print_region_mismatch_summary(cells, tol: float = 1e-3):
    del tol
    regions = {name: [] for name in ("ID", "OOD-X", "OOD-PQ", "OOD-X+PQ")}
    for key, stats in cells.items():
        regions[_region_name(key)].extend(float(value) for value in stats["mismatch_values"])
    print("\n===== Region Mismatch Summary =====")
    print("Region & Samples & Avg. mismatch & Worst mismatch & Success rate (<0.1) & Success rate (<0.01) & Success rate (<0.001) ")
    for name, values in regions.items():
        if not values:
            print(f"{name} & 0 & N/A & N/A & N/A & N/A & N/A ")
            continue
        arr = np.asarray(values, dtype=np.float64)
        print(
            f"{name} & {len(values)} & {arr.mean():.6e} & {arr.max():.6e} & "
            f"{np.mean(arr < 1e-1):.6f} & {np.mean(arr < 1e-2):.6f} & {np.mean(arr < 1e-3):.6f} "
        )


def write_heatmap_cell_csv(cells, output_csv):
    write_heatmap_csv(cells, output_csv)


def write_ill_mismatch_csv(mismatch_values, output_csv):
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "mismatch"])
        for sample_idx, mismatch in enumerate(mismatch_values, start=1):
            writer.writerow([sample_idx, f"{float(mismatch):.12e}"])
