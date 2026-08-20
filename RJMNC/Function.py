import numpy as np
from Loss import *
from copy import copy
import torch
from collections import defaultdict
import csv
import os
import time

def _clean_zero(x):
    return 0.0 if abs(float(x)) < 5e-8 else float(x)


def _write_proposed_heatmap_csv(cell_stats, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    x_values = sorted({
        round(_clean_zero((float(key[0]) + float(key[1])) / 2.0), 10)
        for key in cell_stats.keys()
    })
    y_values = sorted({
        round(_clean_zero((float(key[2]) + float(key[3])) / 2.0), 10)
        for key in cell_stats.keys()
    })

    value_by_xy = {}
    for x_low, x_high, pq_low, pq_high in cell_stats.keys():
        stat = cell_stats[(x_low, x_high, pq_low, pq_high)]
        x_center = round(_clean_zero((float(x_low) + float(x_high)) / 2.0), 10)
        y_center = round(_clean_zero((float(pq_low) + float(pq_high)) / 2.0), 10)
        value_by_xy[(x_center, y_center)] = max(float(stat["p_max"]), float(stat["q_max"]))

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["PQ\\X"] + [f"{x:.1f}" for x in x_values])
        for y in y_values:
            writer.writerow([f"{y:.1f}"] + [value_by_xy.get((x, y), "") for x in x_values])


def _write_ill_mismatch_csv(mismatch_values, output_csv):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "mismatch"])
        for sample_idx, mismatch in enumerate(mismatch_values, start=1):
            writer.writerow([sample_idx, f"{float(mismatch):.12e}"])


def get_scheduler(optimizer, patience):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=patience,
        threshold=1e-6,
        threshold_mode="abs",
        min_lr=1e-6
    )

def _new_cell_stat():
    return {
        "count": 0, "valid_label_count": 0, "invalid_label_count": 0, "iter_sum": 0.0, "angle_max": 0.0, "volt_max": 0.0, "p_max": 0.0, "q_max": 0.0,
        "angle_sample_sum": 0.0, "angle_sample_sq_sum": 0.0, "angle_sample_count": 0,
        "volt_sample_sum": 0.0, "volt_sample_sq_sum": 0.0, "volt_sample_count": 0,
        "p_sample_sum": 0.0, "p_sample_sq_sum": 0.0, "p_sample_count": 0,
        "q_sample_sum": 0.0, "q_sample_sq_sum": 0.0, "q_sample_count": 0,
        "p_mae_sum": 0.0, "q_mae_sum": 0.0, "p_mse_sum": 0.0, "q_mse_sum": 0.0, "p_rmse_sum": 0.0, "q_rmse_sum": 0.0,
        "mismatch_values": [],
        "stop_reason_cnt": {"converged": 0, "max_blocks": 0, "numerical_failure": 0},
    }


def _mean_std_from_samples(stat, prefix):
    count = int(stat[f"{prefix}_sample_count"])
    if count <= 0:
        return None, None
    mean = float(stat[f"{prefix}_sample_sum"]) / count
    mean_sq = float(stat[f"{prefix}_sample_sq_sum"]) / count
    std = float(np.sqrt(max(mean_sq - mean * mean, 0.0)))
    return mean, std


def _axis_in_train_space(low, high):
    return float(low) >= -1.0 - 1e-9 and float(high) <= 1.0 + 1e-9


def _region_name(x_low, x_high, pq_low, pq_high):
    x_id = _axis_in_train_space(x_low, x_high)
    pq_id = _axis_in_train_space(pq_low, pq_high)
    if x_id and pq_id:
        return "ID"
    if (not x_id) and pq_id:
        return "OOD-X"
    if x_id and (not pq_id):
        return "OOD-PQ"
    return "OOD-X+PQ"


def _log_proposed_region_mismatch_summary(cell_stats, log_name):
    regions = {"ID": [], "OOD-X": [], "OOD-PQ": [], "OOD-X+PQ": []}
    for x_low, x_high, pq_low, pq_high in sorted(cell_stats.keys()):
        region = _region_name(x_low, x_high, pq_low, pq_high)
        regions[region].extend(float(v) for v in cell_stats[(x_low, x_high, pq_low, pq_high)]["mismatch_values"])

    log_print("\n===== Region Mismatch Summary =====", log_name)
    log_print("Region & Samples & Avg. mismatch & Worst mismatch & Success rate (<0.1) & Success rate (<0.01) & Success rate (<0.001) & Success rate (<5E-4) & Success rate (<1E-4)", log_name)
    for region in ("ID", "OOD-X", "OOD-PQ", "OOD-X+PQ"):
        values = regions[region]
        if not values:
            log_print(f"{region} & 0 & N/A & N/A & N/A & N/A & N/A & N/A & N/A", log_name)
            continue
        arr = np.asarray(values, dtype=np.float64)
        success_1e_1 = float(np.mean(arr < 1e-1))
        success_1e_2 = float(np.mean(arr < 1e-2))
        success_1e_3 = float(np.mean(arr < 1e-3))
        success_5e_4 = float(np.mean(arr < 5e-4))
        success_1e_4 = float(np.mean(arr < 1e-4))
        log_print(
            f"{region} & {len(values)} & {float(arr.mean()):.6e} & {float(arr.max()):.6e} & "
            f"{success_1e_1:.6f} & {success_1e_2:.6f} & {success_1e_3:.6f} & "
            f"{success_5e_4:.6f} & {success_1e_4:.6f}",
            log_name,
        )

def _batch_max_mismatch_from_state(state, node_mean, node_std):
    num_graphs = int(state.num_graphs)
    non_slack = int(state.x.shape[0] // num_graphs)

    x_raw = state.x.view(num_graphs, non_slack, -1) * node_std.view(1, 1, -1) + node_mean.view(1, 1, -1)
    masks = state.masks.view(num_graphs, non_slack)
    pq_mask = (masks == 1)

    delta_p = x_raw[:, :, 2]
    delta_q = x_raw[:, :, 3]

    max_dp = delta_p.abs().amax(dim=1)
    q_abs = delta_q.abs()
    max_dq = torch.where(
        pq_mask.any(dim=1),
        torch.where(pq_mask, q_abs, torch.zeros_like(q_abs)).amax(dim=1),
        torch.zeros(num_graphs, device=state.x.device, dtype=state.x.dtype),
    )
    return torch.maximum(max_dp, max_dq)


def _select_fixed_size_graphs(state, keep_graphs, non_slack):
    """Compact a homogeneous PyG batch to the requested graph indices.

    RJMNC datasets contain a fixed number of nodes per graph.  Compacting the
    batch lets later blocks avoid GAT and power-flow work for samples that have
    already met the paper's convergence threshold.
    """
    old_num_graphs = int(state.num_graphs)
    keep_graphs = keep_graphs.to(device=state.x.device, dtype=torch.long)
    new_num_graphs = int(keep_graphs.numel())
    if new_num_graphs == old_num_graphs:
        return state
    if new_num_graphs < 1:
        raise ValueError("Cannot construct an empty active graph batch.")

    selected = copy(state)
    selected.x = state.x.view(old_num_graphs, non_slack, -1)[keep_graphs].reshape(
        new_num_graphs * non_slack, -1
    )
    selected.masks = state.masks.view(old_num_graphs, non_slack)[keep_graphs].reshape(-1)
    if getattr(state, "y", None) is not None and state.y.shape[0] == old_num_graphs * non_slack:
        selected.y = state.y.view(old_num_graphs, non_slack, -1)[keep_graphs].reshape(
            new_num_graphs * non_slack, -1
        )

    edge_graph = state.batch[state.edge_index[0]]
    graph_remap = torch.full(
        (old_num_graphs,), -1, device=state.x.device, dtype=torch.long
    )
    graph_remap[keep_graphs] = torch.arange(
        new_num_graphs, device=state.x.device, dtype=torch.long
    )
    edge_keep = graph_remap[edge_graph] >= 0
    kept_edge_graph = edge_graph[edge_keep]
    local_edge_index = state.edge_index[:, edge_keep] - kept_edge_graph.view(1, -1) * non_slack
    new_edge_graph = graph_remap[kept_edge_graph]
    selected.edge_index = local_edge_index + new_edge_graph.view(1, -1) * non_slack
    selected.edge_attr = state.edge_attr[edge_keep]
    selected.batch = torch.arange(
        new_num_graphs, device=state.x.device, dtype=torch.long
    ).repeat_interleave(non_slack)
    selected.ptr = torch.arange(
        new_num_graphs + 1, device=state.x.device, dtype=torch.long
    ) * non_slack
    selected._num_graphs = new_num_graphs
    return selected


def _pinn_calc_power(Gij, Bij, vm, va):
    dtype = vm.dtype
    g = Gij.to(device=vm.device, dtype=dtype)
    b = Bij.to(device=vm.device, dtype=dtype)
    theta = va.unsqueeze(2) - va.unsqueeze(1)
    vprod = vm.unsqueeze(2) * vm.unsqueeze(1)
    p_calc = torch.sum(vprod * (g.unsqueeze(0) * torch.cos(theta) + b.unsqueeze(0) * torch.sin(theta)), dim=2)
    q_calc = torch.sum(vprod * (g.unsqueeze(0) * torch.sin(theta) - b.unsqueeze(0) * torch.cos(theta)), dim=2)
    return p_calc, q_calc


def _pinn_polar_jacobian(Gij, Bij, vm, va, non, pq):
    dtype = vm.dtype
    device = vm.device
    g = Gij.to(device=device, dtype=dtype)
    b = Bij.to(device=device, dtype=dtype)
    p_calc, q_calc = _pinn_calc_power(g, b, vm, va)
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
        row_for_pq = torch.nonzero(non.unsqueeze(1).eq(pq.unsqueeze(0)), as_tuple=False)[:, 0]
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


def _pinn_mismatch_values_from_voltage(Gij, Bij, vm, va, p_spec, q_spec, non, pq):
    p_calc, q_calc = _pinn_calc_power(Gij, Bij, vm, va)
    dp = p_spec - p_calc
    dq = q_spec - q_calc
    p_batch = dp[:, non]
    q_batch = dq[:, pq]
    max_p = p_batch.abs().amax(dim=1) if p_batch.numel() > 0 else torch.zeros(vm.shape[0], device=vm.device, dtype=vm.dtype)
    max_q = q_batch.abs().amax(dim=1) if q_batch.numel() > 0 else torch.zeros(vm.shape[0], device=vm.device, dtype=vm.dtype)
    return torch.maximum(max_p, max_q)


def _pinn_nr_refine_until_from_best_x(best_x, masks, Gij, Bij, slack_node, node_mean, node_std, pf_cache, max_steps=20, tol=1e-3):
    device = best_x.device
    dtype = best_x.dtype
    num_graphs, non_slack, _ = best_x.shape
    total_nodes = int(pf_cache["total_nodes"])
    non = pf_cache["non_to_full"].to(device=device, dtype=torch.long)
    slack_idx = int(pf_cache["slack_idx"])
    pq_mask = masks == 1
    pq = non[pq_mask[0]]

    x_raw = best_x * node_std.view(1, 1, -1) + node_mean.view(1, 1, -1)
    vm = torch.zeros(num_graphs, total_nodes, device=device, dtype=dtype)
    va = torch.zeros_like(vm)
    p_spec = torch.zeros_like(vm)
    q_spec = torch.zeros_like(vm)

    vm[:, non] = x_raw[:, :, 5]
    va[:, non] = x_raw[:, :, 4]
    vm[:, slack_idx] = slack_node[2].to(device=device, dtype=dtype)
    va[:, slack_idx] = slack_node[1].to(device=device, dtype=dtype)
    p_spec[:, non] = x_raw[:, :, 0] + x_raw[:, :, 2]
    q_spec[:, non] = x_raw[:, :, 1] + x_raw[:, :, 3]

    finite = torch.isfinite(vm).all(dim=1) & torch.isfinite(va).all(dim=1)
    init_mismatch = _pinn_mismatch_values_from_voltage(Gij, Bij, vm, va, p_spec, q_spec, non, pq)
    final_mismatch = init_mismatch.clone()
    success = finite & (init_mismatch < tol)
    iterations = torch.zeros(num_graphs, dtype=torch.long, device=device)
    active = finite & (~success)
    n_non = int(non.numel())
    n_pq = int(pq.numel())
    for step in range(1, max_steps + 1):
        if not bool(active.any().item()):
            break
        p_calc, q_calc = _pinn_calc_power(Gij, Bij, vm, va)
        rhs = torch.cat([(p_spec - p_calc)[:, non], (q_spec - q_calc)[:, pq]], dim=1)
        jac = _pinn_polar_jacobian(Gij, Bij, vm, va, non, pq)
        sol, solve_info = torch.linalg.solve_ex(jac, rhs.unsqueeze(-1))
        delta = sol.squeeze(-1)
        ok = active & (solve_info == 0) & torch.isfinite(delta).all(dim=1)
        if not bool(ok.any().item()):
            break

        ok_idx = torch.where(ok)[0]
        va_sub = va[ok_idx].clone()
        va_sub[:, non] = va_sub[:, non] + delta[ok_idx, :n_non]
        va[ok_idx] = va_sub
        if n_pq:
            vm_sub = vm[ok_idx].clone()
            vm_sub[:, pq] = torch.clamp(vm_sub[:, pq] + delta[ok_idx, n_non:], min=0.2, max=2.5)
            vm[ok_idx] = vm_sub
        vm[:, slack_idx] = slack_node[2].to(device=device, dtype=dtype)
        va[:, slack_idx] = slack_node[1].to(device=device, dtype=dtype)
        mismatch = _pinn_mismatch_values_from_voltage(Gij, Bij, vm, va, p_spec, q_spec, non, pq)
        final_mismatch[ok] = mismatch[ok]
        newly_success = ok & (mismatch < tol)
        iterations[newly_success] = step
        success = success | newly_success
        active = ok & (~success)

    return success, iterations, final_mismatch


@torch.inference_mode()
def eval_convergence_metrics_fast(model, data_loader, device, Gij_GPU, Bij_GPU, slack_node_GPU, node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU, lines_mean_GPU, lines_std_GPU, unroll_steps, pf_cache, tol_mismatch=1e-3, max_blocks=2,):
    if pf_cache is None:
        raise ValueError("eval_convergence_metrics_fast requires pf_cache.")
    if max_blocks < 1:
        raise ValueError("max_blocks must be >= 1.")

    model.eval()
    total_graphs = 0
    conv_counts = [0 for _ in range(max_blocks)]
    residual_sums = [0.0 for _ in range(max_blocks)]
    excess_sums = [0.0 for _ in range(max_blocks)]
    residual_maxes = [0.0 for _ in range(max_blocks)]

    for data in data_loader:
        data = data.to(device, non_blocking=True)
        state = data.clone()
        num_graphs = int(state.num_graphs)
        total_graphs += num_graphs

        for block_idx in range(max_blocks):
            for inner_step in range(unroll_steps):
                out = model(state, step=inner_step)
                state = update_state_differentiable_sparse(
                    state, out, Gij_GPU, Bij_GPU, slack_node_GPU,
                    node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU,
                    lines_mean_GPU, lines_std_GPU, pf_cache,
                    update_edge_attr=not (
                        block_idx == max_blocks - 1 and inner_step == unroll_steps - 1
                    ),
                )

            max_mismatch = _batch_max_mismatch_from_state(state, node_mean_GPU, node_std_GPU)
            conv_counts[block_idx] += int((max_mismatch < tol_mismatch).sum().item())
            residual_sums[block_idx] += float(max_mismatch.sum().item())
            excess = torch.relu(max_mismatch / tol_mismatch - 1.0)
            excess_sums[block_idx] += float(excess.sum().item())
            residual_maxes[block_idx] = max(
                residual_maxes[block_idx],
                float(max_mismatch.max().item()),
            )

    if total_graphs == 0:
        return {
            "conv_rates": [0.0 for _ in range(max_blocks)],
            "residual_means": [float("inf") for _ in range(max_blocks)],
            "excess_means": [float("inf") for _ in range(max_blocks)],
            "residual_maxes": [float("inf") for _ in range(max_blocks)],
            "total_graphs": 0,
        }

    conv_rates = [count / total_graphs for count in conv_counts]
    residual_means = [res_sum / total_graphs for res_sum in residual_sums]
    excess_means = [excess_sum / total_graphs for excess_sum in excess_sums]
    return {
        "conv_rates": conv_rates,
        "residual_means": residual_means,
        "excess_means": excess_means,
        "residual_maxes": residual_maxes,
        "total_graphs": total_graphs,
    }

def test_unrolls_batch(model, test_loader, device, Gij_GPU, Bij_GPU, slack_node_GPU, node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU, lines_mean_GPU, lines_std_GPU, unroll_steps, max_blocks=20, tol_mismatch=1e-3, print_samples=50, pf_cache=None, log_name = "", node_chose = ""):

    if pf_cache is None:
        raise ValueError("test_unrolls_batch now requires pf_cache when using sparse update.")
    if max_blocks < 1:
        raise ValueError("max_blocks must be >= 1.")
    model.eval()
    cell_stats = defaultdict(_new_cell_stat)
    stop_reason_cnt = {"converged": 0, "max_blocks": 0, "numerical_failure": 0}

    sample_count = 0
    compute_time = 0.0
    cuda_timing_events = []
    executed_sample_blocks = 0
    possible_sample_blocks = 0
    with torch.inference_mode():
        for data in test_loader:
            data = data.to(device)

            state = data.clone()
            num_graphs = int(state.num_graphs)
            non_slack = int(state.x.shape[0] // num_graphs)
            feat_dim = int(state.x.shape[1])

            if not hasattr(state, "level"):
                raise ValueError("test_unrolls_batch requires `data.level` heatmap metadata.")
            if int(state.level.numel()) != num_graphs * 6:
                raise ValueError(f"Invalid `data.level` shape: expected {num_graphs * 6} elements, got {state.level.numel()}.")

            level_meta = state.level.view(num_graphs, 6).to(dtype=state.x.dtype)
            x_low = level_meta[:, 0]
            x_high = level_meta[:, 1]
            pq_low = level_meta[:, 2]
            pq_high = level_meta[:, 3]
            x_signed = level_meta[:, 4]
            pq_signed = level_meta[:, 5]

            if hasattr(state, "valid_label"):
                valid_label = state.valid_label.view(-1).bool()
            else:
                valid_label = torch.ones(num_graphs, device=device, dtype=torch.bool)

            sample_ids = list(range(sample_count + 1, sample_count + num_graphs + 1))
            sample_count += num_graphs
            print_mask = torch.tensor([sid <= print_samples for sid in sample_ids], device=device, dtype=torch.bool)
            sample_logs = [[] for _ in range(num_graphs)]

            masks = state.masks.view(num_graphs, non_slack)
            pq_mask = (masks == 1)

            init_x_raw = state.x.view(num_graphs, non_slack, feat_dim) * node_std_GPU.view(1, 1, -1) + node_mean_GPU.view(1, 1, -1)
            y_raw = state.y.view(num_graphs, non_slack, -1) * output_std_GPU.view(1, 1, -1) + output_mean_GPU.view(1, 1, -1)

            angle_true = init_x_raw[:, :, 4] + y_raw[:, :, 0]
            voltage_true = init_x_raw[:, :, 5].clone()
            voltage_true[pq_mask] = voltage_true[pq_mask] + y_raw[:, :, 1][pq_mask]

            delta_p0 = init_x_raw[:, :, 2]
            delta_q0 = init_x_raw[:, :, 3]
            max_dp0 = delta_p0.abs().amax(dim=1)
            q_abs0 = delta_q0.abs()
            max_dq0 = torch.where(pq_mask.any(dim=1), torch.where(pq_mask, q_abs0, torch.zeros_like(q_abs0)).amax(dim=1), torch.zeros(num_graphs, device=device, dtype=state.x.dtype),)
            max_mismatch0 = torch.maximum(max_dp0, max_dq0)

            max_err_ang0 = (init_x_raw[:, :, 4] - angle_true).abs().amax(dim=1)
            max_err_vol0 = torch.where(
                pq_mask.any(dim=1),
                torch.where(pq_mask, (init_x_raw[:, :, 5] - voltage_true).abs(), torch.zeros_like(init_x_raw[:, :, 5]),).amax(dim=1),
                torch.zeros(num_graphs, device=device, dtype=state.x.dtype),
            )

            for g in range(num_graphs):
                if print_mask[g]:
                    true_err_ang_s = f"{max_err_ang0[g].item():.3e}" if valid_label[g] else "N/A"
                    true_err_vol_s = f"{max_err_vol0[g].item():.3e}" if valid_label[g] else "N/A"
                    sample_logs[g].append(
                        f"[sample {sample_ids[g]:03d}] iter 00: "
                        f"max_mismatchPQ={max_mismatch0[g].item():.3e}, NN_max_dAng=0.000e+00, NN_max_dV=0.000e+00  "
                        f"True_errAng={true_err_ang_s}, True_errV={true_err_vol_s}"
                    )

            best_mismatch = max_mismatch0.clone()
            best_block = torch.zeros(num_graphs, device=device, dtype=torch.long)
            best_x = state.x.view(num_graphs, non_slack, feat_dim).clone()
            active_ids = torch.arange(num_graphs, device=device, dtype=torch.long)
            stop_reason = ["max_blocks"] * num_graphs
            possible_sample_blocks += num_graphs * max_blocks

            if device.type == "cuda":
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end = torch.cuda.Event(enable_timing=True)
                compute_start.record()
            else:
                block_compute_start = time.perf_counter()

            for block in range(1, max_blocks + 1):
                active_count = int(state.num_graphs)
                executed_sample_blocks += active_count
                active_pq_mask = state.masks.view(active_count, non_slack) == 1
                block_delta_phys = torch.zeros(
                    (active_count, non_slack, 2), device=device, dtype=state.x.dtype
                )

                for inner_step in range(unroll_steps):
                    out = model(state, step=inner_step)
                    out_reshaped = out.view(active_count, non_slack, -1)
                    out_phys = out_reshaped * output_std_GPU.view(1, 1, -1) + output_mean_GPU.view(1, 1, -1)
                    block_delta_phys[:, :, 0] = block_delta_phys[:, :, 0] + out_phys[:, :, 0]
                    block_delta_phys[:, :, 1] = block_delta_phys[:, :, 1] + torch.where(active_pq_mask, out_phys[:, :, 1], torch.zeros_like(out_phys[:, :, 1]))
                    state = update_state_differentiable_sparse(
                        state, out, Gij_GPU, Bij_GPU, slack_node_GPU,
                        node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU,
                        lines_mean_GPU, lines_std_GPU, pf_cache,
                        update_edge_attr=not (
                            block == max_blocks and inner_step == unroll_steps - 1
                        ),
                    )

                state_x = state.x.view(active_count, non_slack, feat_dim)
                x_raw = state_x * node_std_GPU.view(1, 1, -1) + node_mean_GPU.view(1, 1, -1)
                delta_p = x_raw[:, :, 2]
                delta_q = x_raw[:, :, 3]

                max_dp = delta_p.abs().amax(dim=1)
                q_abs = delta_q.abs()
                max_dq = torch.where(
                    active_pq_mask.any(dim=1),
                    torch.where(active_pq_mask, q_abs, torch.zeros_like(q_abs)).amax(dim=1),
                    torch.zeros(active_count, device=device, dtype=state.x.dtype),
                )
                max_mismatch = torch.maximum(max_dp, max_dq)

                angle_est = x_raw[:, :, 4]
                voltage_est = x_raw[:, :, 5]
                max_err_ang = (angle_est - angle_true[active_ids]).abs().amax(dim=1)
                max_err_vol = torch.where(
                    active_pq_mask.any(dim=1),
                    torch.where(active_pq_mask, (voltage_est - voltage_true[active_ids]).abs(), torch.zeros_like(voltage_est)).amax(dim=1),
                    torch.zeros(active_count, device=device, dtype=state.x.dtype),
                )

                block_max_dang = block_delta_phys[:, :, 0].abs().amax(dim=1)
                block_max_dv = torch.where(
                    active_pq_mask.any(dim=1),
                    torch.where(active_pq_mask, block_delta_phys[:, :, 1].abs(), torch.zeros_like(block_delta_phys[:, :, 1])).amax(dim=1),
                    torch.zeros(active_count, device=device, dtype=state.x.dtype),
                )

                finite = torch.isfinite(max_mismatch) & torch.isfinite(
                    state_x.reshape(active_count, -1)
                ).all(dim=1)
                improved = finite & (max_mismatch < best_mismatch[active_ids])
                improved_ids = active_ids[improved]
                best_mismatch[improved_ids] = max_mismatch[improved]
                best_block[improved_ids] = block
                best_x[improved_ids] = state_x[improved]

                converged = finite & (max_mismatch <= tol_mismatch)
                # The paper returns x^{k,T} when the threshold is reached.
                # Best-so-far fallback applies only when no block converges.
                converged_ids = active_ids[converged]
                best_mismatch[converged_ids] = max_mismatch[converged]
                best_block[converged_ids] = block
                best_x[converged_ids] = state_x[converged]
                numerical_failure = ~finite
                for original_g in active_ids[converged].tolist():
                    stop_reason[original_g] = "converged"
                for original_g in active_ids[numerical_failure].tolist():
                    stop_reason[original_g] = "numerical_failure"

                printed_local = torch.where(print_mask[active_ids])[0]
                printed_pairs = torch.stack(
                    [printed_local, active_ids[printed_local]], dim=1
                ).tolist()
                for local_g, original_g in printed_pairs:
                    true_err_ang_s = f"{max_err_ang[local_g].item():.3e}" if valid_label[original_g] else "N/A"
                    true_err_vol_s = f"{max_err_vol[local_g].item():.3e}" if valid_label[original_g] else "N/A"
                    sample_logs[original_g].append(
                        f"[sample {sample_ids[original_g]:03d}] iter {block:02d}: "
                        f"max_mismatchPQ={max_mismatch[local_g].item():.3e}, "
                        f"NN_max_dAng={block_max_dang[local_g].item():.3e}, "
                        f"NN_max_dV={block_max_dv[local_g].item():.3e}  "
                        f"True_errAng={true_err_ang_s}, True_errV={true_err_vol_s}"
                    )

                keep_local = torch.where(finite & (~converged))[0]
                if int(keep_local.numel()) == 0:
                    break
                if block < max_blocks:
                    state = _select_fixed_size_graphs(state, keep_local, non_slack)
                    active_ids = active_ids[keep_local]

            if device.type == "cuda":
                compute_end.record()
                cuda_timing_events.append((compute_start, compute_end))
            else:
                compute_time += time.perf_counter() - block_compute_start

            best_x_raw = best_x * node_std_GPU.view(1, 1, -1) + node_mean_GPU.view(1, 1, -1)
            best_angle = best_x_raw[:, :, 4]
            best_voltage = best_x_raw[:, :, 5]
            best_dp = best_x_raw[:, :, 2]
            best_dq = best_x_raw[:, :, 3]

            angle_max = (best_angle - angle_true).abs().amax(dim=1)
            volt_max = torch.where(
                pq_mask.any(dim=1),
                torch.where(pq_mask, (best_voltage - voltage_true).abs(), torch.zeros_like(best_voltage)).amax(dim=1),
                torch.zeros(num_graphs, device=device, dtype=state.x.dtype),
            )

            p_abs = best_dp.abs()
            q_abs = best_dq.abs()
            max_p = p_abs.amax(dim=1)
            max_q = torch.where(
                pq_mask.any(dim=1),
                torch.where(pq_mask, q_abs, torch.zeros_like(q_abs)).amax(dim=1),
                torch.zeros(num_graphs, device=device, dtype=state.x.dtype),
            )

            p_mae = p_abs.mean(dim=1)
            p_mse = (best_dp ** 2).mean(dim=1)
            p_rmse = torch.sqrt(p_mse)

            pq_cnt = pq_mask.sum(dim=1).clamp_min(1).to(dtype=state.x.dtype)
            q_mae = torch.where(pq_mask, q_abs, torch.zeros_like(q_abs)).sum(dim=1) / pq_cnt
            q_mse = torch.where(pq_mask, best_dq ** 2, torch.zeros_like(best_dq)).sum(dim=1) / pq_cnt
            q_rmse = torch.sqrt(q_mse)

            for g in range(num_graphs):
                key = (
                    round(float(x_low[g].item()), 10),
                    round(float(x_high[g].item()), 10),
                    round(float(pq_low[g].item()), 10),
                    round(float(pq_high[g].item()), 10),
                )
                stat = cell_stats[key]
                stat["count"] += 1
                stat["iter_sum"] += float(best_block[g].item())
                sample_p_max = float(max_p[g].item())
                sample_q_max = float(max_q[g].item())
                sample_mismatch = max(sample_p_max, sample_q_max)
                stat["p_max"] = max(stat["p_max"], sample_p_max)
                stat["q_max"] = max(stat["q_max"], sample_q_max)
                stat["p_sample_sum"] += sample_p_max
                stat["p_sample_sq_sum"] += sample_p_max * sample_p_max
                stat["p_sample_count"] += 1
                stat["q_sample_sum"] += sample_q_max
                stat["q_sample_sq_sum"] += sample_q_max * sample_q_max
                stat["q_sample_count"] += 1
                stat["p_mae_sum"] += float(p_mae[g].item())
                stat["q_mae_sum"] += float(q_mae[g].item())
                stat["p_mse_sum"] += float(p_mse[g].item())
                stat["q_mse_sum"] += float(q_mse[g].item())
                stat["p_rmse_sum"] += float(p_rmse[g].item())
                stat["q_rmse_sum"] += float(q_rmse[g].item())
                stat["mismatch_values"].append(sample_mismatch)
                stat["stop_reason_cnt"][stop_reason[g]] += 1
                stop_reason_cnt[stop_reason[g]] += 1

                if valid_label[g]:
                    sample_angle_max = float(angle_max[g].item())
                    sample_volt_max = float(volt_max[g].item())
                    stat["valid_label_count"] += 1
                    stat["angle_max"] = max(stat["angle_max"], sample_angle_max)
                    stat["volt_max"] = max(stat["volt_max"], sample_volt_max)
                    stat["angle_sample_sum"] += sample_angle_max
                    stat["angle_sample_sq_sum"] += sample_angle_max * sample_angle_max
                    stat["angle_sample_count"] += 1
                    stat["volt_sample_sum"] += sample_volt_max
                    stat["volt_sample_sq_sum"] += sample_volt_max * sample_volt_max
                    stat["volt_sample_count"] += 1
                else:
                    stat["invalid_label_count"] += 1

                if print_mask[g]:
                    G, E = "\033[1;32m", "\033[0m"
                    sample_logs[g].append(
                        f"[sample {sample_ids[g]:03d}] >>> stop_reason={stop_reason[g]}, "
                        f"used_iter={int(best_block[g].item())}, "
                        f"max_PQ_mismatch={best_mismatch[g].item():.3e}"
                    )
                    sample_logs[g].append(
                        f"{G}===== Test Sample Heatmap Cell: "
                        f"X=({x_low[g].item():.2f},{x_high[g].item():.2f}), "
                        f"PQ=({pq_low[g].item():.2f},{pq_high[g].item():.2f}), "
                        f"x_signed={x_signed[g].item():.6f}, pq_signed={pq_signed[g].item():.6f}, "
                        f"valid_label={int(valid_label[g].item())} ====={E}"
                    )
                    sample_logs[g].append("---- Error Metrics (Best State) ----")

                    angle_str = f"{float(angle_max[g].item()):.6e}" if valid_label[g] else "N/A"
                    volt_str = f"{float(volt_max[g].item()):.6e}" if valid_label[g] else "N/A"
                    sample_logs[g].append(f"{'Angel_MAX':<12}: {angle_str}")
                    sample_logs[g].append(f"{'Voltage_MAX':<12}: {volt_str}")
                    sample_logs[g].append(f"{'P_MAX':<12}: {float(max_p[g].item()):.6e}")
                    sample_logs[g].append(f"{'Q_MAX':<12}: {float(max_q[g].item()):.6e}")
                    sample_logs[g].append(f"{'P_MAE':<12}: {float(p_mae[g].item()):.6e}")
                    sample_logs[g].append(f"{'Q_MAE':<12}: {float(q_mae[g].item()):.6e}")
                    sample_logs[g].append(f"{'P_MSE':<12}: {float(p_mse[g].item()):.6e}")
                    sample_logs[g].append(f"{'Q_MSE':<12}: {float(q_mse[g].item()):.6e}")
                    sample_logs[g].append(f"{'P_RMSE':<12}: {float(p_rmse[g].item()):.6e}")
                    sample_logs[g].append(f"{'Q_RMSE':<12}: {float(q_rmse[g].item()):.6e}")
                    sample_logs[g].append("")

            for g in range(num_graphs):
                if print_mask[g]:
                    for line in sample_logs[g]:
                        log_print(line,log_name)



    log_print("\n===== Batch Full-Test Summary By Heatmap Cell =====",log_name)
    for key in sorted(cell_stats.keys()):
        x_l, x_h, pq_l, pq_h = key
        stat = cell_stats[key]
        cnt = stat["count"]
        valid_cnt = stat["valid_label_count"]
        invalid_cnt = stat["invalid_label_count"]
        angle_str = f"{stat['angle_max']:.6e}" if valid_cnt > 0 else "N/A"
        volt_str = f"{stat['volt_max']:.6e}" if valid_cnt > 0 else "N/A"
        angle_mean, angle_std = _mean_std_from_samples(stat, "angle")
        volt_mean, volt_std = _mean_std_from_samples(stat, "volt")
        p_mean, p_std = _mean_std_from_samples(stat, "p")
        q_mean, q_std = _mean_std_from_samples(stat, "q")
        angle_dist_str = f"{angle_mean:.6e}±{angle_std:.6e}" if angle_mean is not None else "N/A"
        volt_dist_str = f"{volt_mean:.6e}±{volt_std:.6e}" if volt_mean is not None else "N/A"
        p_dist_str = f"{p_mean:.6e}±{p_std:.6e}" if p_mean is not None else "N/A"
        q_dist_str = f"{q_mean:.6e}±{q_std:.6e}" if q_mean is not None else "N/A"
        mismatch_arr = np.asarray(stat["mismatch_values"], dtype=np.float64)
        success_1e_1 = float(np.mean(mismatch_arr < 1e-1)) if mismatch_arr.size else 0.0
        success_1e_2 = float(np.mean(mismatch_arr < 1e-2)) if mismatch_arr.size else 0.0
        success_1e_3 = float(np.mean(mismatch_arr < 1e-3)) if mismatch_arr.size else 0.0
        log_print(
            f"X=[{x_l:.1f},{x_h:.1f}], PQ=[{pq_l:.1f},{pq_h:.1f}]: "
            f"count:{cnt}, "
            # f"non-ill:{valid_cnt}, "
            f"ill:{invalid_cnt}, "
            f"avg_iter:{stat['iter_sum'] / cnt:.2f}, "
            f"Ang_MAX:{angle_str}, "
            f"Vol_MAX:{volt_str}, "
            f"Ang_mean±std:{angle_dist_str}, "
            f"Vol_mean±std:{volt_dist_str}, "
            f"P_MAX:{stat['p_max']:.3e}, "
            f"Q_MAX:{stat['q_max']:.3e}, "
            f"P_mean±std:{p_dist_str}, "
            f"Q_mean±std:{q_dist_str}, "
            # f"P_MAE:{stat['p_mae_sum'] / cnt:.3e}, "
            # f"Q_MAE:{stat['q_mae_sum'] / cnt:.3e}, "
            # f"P_MSE:{stat['p_mse_sum'] / cnt:.3e}, "
            # f"Q_MSE:{stat['q_mse_sum'] / cnt:.3e}, "
            f"P_RMSE:{stat['p_rmse_sum'] / cnt:.3e}, "
            f"Q_RMSE:{stat['q_rmse_sum'] / cnt:.3e}, "
            f"Success rate (<0.1):{success_1e_1:.6f}, "
            f"Success rate (<0.01):{success_1e_2:.6f}, "
            f"Success rate (<0.001):{success_1e_3:.6f}"
            ,log_name)
    paths = node_chose + ".csv"
    csv_path = os.path.join(os.path.dirname(__file__), "ckpt", paths)
    _write_proposed_heatmap_csv(cell_stats, csv_path)
    log_print(f"Heatmap CSV saved: {csv_path}", log_name)

    _log_proposed_region_mismatch_summary(cell_stats, log_name)

    log_print("\n===== Stop Reason Breakdown =====",log_name)
    for reason, cnt in stop_reason_cnt.items():
        log_print(f"  {reason:12s}: {cnt}",log_name)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        compute_time = sum(
            start.elapsed_time(end) for start, end in cuda_timing_events
        ) / 1000.0
    test_elapsed = compute_time
    saved_sample_blocks = possible_sample_blocks - executed_sample_blocks
    log_print(
        f"\nActive sample-block evaluations  : {executed_sample_blocks}/{possible_sample_blocks} "
        f"(skipped {saved_sample_blocks})",
        log_name,
    )
    log_print(f"Test iterative compute time      : {test_elapsed:.6f} s", log_name)
    return {
        "cell_stats": dict(cell_stats),
        "stop_reason_cnt": stop_reason_cnt,
        "test_elapsed_sec": test_elapsed,
        "executed_sample_blocks": executed_sample_blocks,
        "skipped_sample_blocks": saved_sample_blocks,
    }


def test_ill_conditioned_batch(model, test_loader, device, Gij_GPU, Bij_GPU, slack_node_GPU, node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU, lines_mean_GPU, lines_std_GPU, unroll_steps, max_blocks=20, tol_mismatch=1e-3, pf_cache=None, log_name="", mismatch_csv_path=None):
    if pf_cache is None:
        raise ValueError("test_ill_conditioned_batch requires pf_cache when using sparse update.")
    if max_blocks < 1:
        raise ValueError("max_blocks must be >= 1.")
    if hasattr(test_loader, "dataset") and len(test_loader.dataset) == 0:
        log_print("\n===== Ill-Conditioned Test Summary =====", log_name)
        log_print("Samples & Maximal mismatch & Minimal mismatch & Avg. mismatch & Success rate (<0.1) & Success rate (<0.01) & Success rate (<0.001) & Avg. mismatch With N-R PF refine & Success rate With N-R PF refine & Average N-R PF Iteration time", log_name)
        log_print("0 & N/A & N/A & N/A & N/A & N/A & N/A & N/A & N/A & N/A", log_name)
        if mismatch_csv_path is not None:
            _write_ill_mismatch_csv([], mismatch_csv_path)
        return {"samples": 0, "max_mismatch": None, "min_mismatch": None, "avg_mismatch": None, "success_rate_1e_1": None, "success_rate_1e_2": None, "success_rate_1e_3": None, "nr_refined_avg_mismatch": None, "nr_refined_success_rate_1e_3": None, "nr_refined_avg_iter_1e_3": None,}
    model.eval()
    mismatch_values = []
    nr_refined_mismatch_values = []
    nr_refined_success_values = []
    nr_refined_success_iters = []
    with torch.inference_mode():
        for data in test_loader:
            data = data.to(device)
            state = data.clone()
            num_graphs = int(state.num_graphs)
            non_slack = int(state.x.shape[0] // num_graphs)
            feat_dim = int(state.x.shape[1])

            masks = state.masks.view(num_graphs, non_slack)
            pq_mask = (masks == 1)

            x_raw0 = state.x.view(num_graphs, non_slack, feat_dim) * node_std_GPU.view(1, 1, -1) + node_mean_GPU.view(1, 1, -1)
            max_dp0 = x_raw0[:, :, 2].abs().amax(dim=1)
            q_abs0 = x_raw0[:, :, 3].abs()
            max_dq0 = torch.where(
                pq_mask.any(dim=1),
                torch.where(pq_mask, q_abs0, torch.zeros_like(q_abs0)).amax(dim=1),
                torch.zeros(num_graphs, device=device, dtype=state.x.dtype),
            )
            best_mismatch = torch.maximum(max_dp0, max_dq0)
            best_x = state.x.view(num_graphs, non_slack, feat_dim).clone()
            active_ids = torch.arange(num_graphs, device=device, dtype=torch.long)

            for block in range(1, max_blocks + 1):
                active_count = int(state.num_graphs)
                active_pq_mask = state.masks.view(active_count, non_slack) == 1
                for inner_step in range(unroll_steps):
                    out = model(state, step=inner_step)
                    state = update_state_differentiable_sparse(
                        state, out, Gij_GPU, Bij_GPU, slack_node_GPU,
                        node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU,
                        lines_mean_GPU, lines_std_GPU, pf_cache,
                        update_edge_attr=not (
                            block == max_blocks and inner_step == unroll_steps - 1
                        ),
                    )

                state_x = state.x.view(active_count, non_slack, feat_dim)
                x_raw = state_x * node_std_GPU.view(1, 1, -1) + node_mean_GPU.view(1, 1, -1)
                max_dp = x_raw[:, :, 2].abs().amax(dim=1)
                q_abs = x_raw[:, :, 3].abs()
                max_dq = torch.where(
                    active_pq_mask.any(dim=1),
                    torch.where(active_pq_mask, q_abs, torch.zeros_like(q_abs)).amax(dim=1),
                    torch.zeros(active_count, device=device, dtype=state.x.dtype),
                )
                max_mismatch = torch.maximum(max_dp, max_dq)

                finite = torch.isfinite(max_mismatch) & torch.isfinite(
                    state_x.reshape(active_count, -1)
                ).all(dim=1)
                improved = finite & (max_mismatch < best_mismatch[active_ids])
                improved_ids = active_ids[improved]
                best_mismatch[improved_ids] = max_mismatch[improved]
                best_x[improved_ids] = state_x[improved]

                converged = finite & (max_mismatch <= tol_mismatch)
                converged_ids = active_ids[converged]
                best_mismatch[converged_ids] = max_mismatch[converged]
                best_x[converged_ids] = state_x[converged]
                keep_local = torch.where(finite & (~converged))[0]
                if int(keep_local.numel()) == 0:
                    break
                if block < max_blocks:
                    state = _select_fixed_size_graphs(state, keep_local, non_slack)
                    active_ids = active_ids[keep_local]

            mismatch_values.extend(best_mismatch.detach().cpu().tolist())
            nr_success, nr_iters, nr_mismatch = _pinn_nr_refine_until_from_best_x(best_x, masks, Gij_GPU, Bij_GPU, slack_node_GPU, node_mean_GPU, node_std_GPU, pf_cache, max_steps=20, tol=1e-5,)
            nr_refined_mismatch_values.extend(nr_mismatch.detach().cpu().tolist())
            nr_refined_success_values.extend(nr_success.to(torch.float32).detach().cpu().tolist())
            if bool(nr_success.any().item()):
                nr_refined_success_iters.extend(nr_iters[nr_success].detach().cpu().tolist())

    arr = np.asarray(mismatch_values, dtype=np.float64)
    nr_mismatch_arr = np.asarray(nr_refined_mismatch_values, dtype=np.float64)
    nr_success_arr = np.asarray(nr_refined_success_values, dtype=np.float64)
    nr_iter_arr = np.asarray(nr_refined_success_iters, dtype=np.float64)
    nr_success_mask = nr_success_arr.astype(bool) if nr_success_arr.size else np.asarray([], dtype=bool)
    if mismatch_csv_path is not None:
        _write_ill_mismatch_csv(arr, mismatch_csv_path)
        log_print(f"Ill-conditioned mismatch CSV saved: {mismatch_csv_path}", log_name)
    max_mismatch = float(np.max(arr)) if arr.size else None
    min_mismatch = float(np.min(arr)) if arr.size else None
    avg_mismatch = float(np.mean(arr)) if arr.size else None
    success_1e_1 = float(np.mean(arr < 1e-1)) if arr.size else None
    success_1e_2 = float(np.mean(arr < 1e-2)) if arr.size else None
    success_1e_3 = float(np.mean(arr < 1e-3)) if arr.size else None
    nr_avg_mismatch = float(np.mean(nr_mismatch_arr[nr_success_mask])) if nr_success_mask.any() else None
    nr_success_1e_3 = float(np.mean(nr_success_arr)) if nr_success_arr.size else None
    nr_avg_iter_1e_3 = float(np.mean(nr_iter_arr)) if nr_iter_arr.size else None
    log_print("\n===== Ill-Conditioned Test Summary =====", log_name)
    log_print("Samples & Maximal mismatch & Minimal mismatch & Avg. mismatch & Success rate (<0.1) & Success rate (<0.01) & Success rate (<0.001) & Avg. mismatch With N-R PF refine & Success rate With N-R PF refine & Average N-R PF Iteration time", log_name)
    max_mismatch_str = "N/A" if max_mismatch is None else f"{max_mismatch:.6e}"
    min_mismatch_str = "N/A" if min_mismatch is None else f"{min_mismatch:.6e}"
    avg_mismatch_str = "N/A" if avg_mismatch is None else f"{avg_mismatch:.6e}"
    nr_avg_mismatch_str = "N/A" if nr_avg_mismatch is None else f"{nr_avg_mismatch:.6e}"
    nr_success_str = "N/A" if nr_success_1e_3 is None else f"{nr_success_1e_3:.6f}"
    nr_avg_iter_str = "N/A" if nr_avg_iter_1e_3 is None else f"{nr_avg_iter_1e_3:.6f}"
    log_print(f"{arr.size} & {max_mismatch_str} & {min_mismatch_str} & {avg_mismatch_str} & {success_1e_1:.6f} & {success_1e_2:.6f} & " f"{success_1e_3:.6f} & {nr_avg_mismatch_str} & {nr_success_str} & {nr_avg_iter_str}", log_name)
    return {"samples": int(arr.size), "max_mismatch": max_mismatch, "min_mismatch": min_mismatch, "avg_mismatch": avg_mismatch, "success_rate_1e_1": success_1e_1, "success_rate_1e_2": success_1e_2, "success_rate_1e_3": success_1e_3, "nr_refined_avg_mismatch": nr_avg_mismatch, "nr_refined_success_rate_1e_3": nr_success_1e_3, "nr_refined_avg_iter_1e_3": nr_avg_iter_1e_3}

def build_pf_sparse_cache(Gij, Bij, slack_node):
    device = Gij.device
    total_nodes = Gij.shape[0]

    slack_idx = int(slack_node[0].item()) - 1
    non_slack = total_nodes - 1

    full_idx = torch.arange(total_nodes, device=device, dtype=torch.long)

    non_to_full = torch.cat([full_idx[:slack_idx], full_idx[slack_idx + 1:]], dim=0)

    full_to_non = torch.full((total_nodes,), -1, device=device, dtype=torch.long)
    full_to_non[non_to_full] = torch.arange(non_slack, device=device, dtype=torch.long)

    nz_mask = (Gij != 0) | (Bij != 0)
    src_full_all, dst_full_all = torch.nonzero(nz_mask, as_tuple=True)

    src_non_all = full_to_non[src_full_all]
    dst_non_all = full_to_non[dst_full_all]

    row_non_mask = src_non_all >= 0

    pf_src_full = src_full_all[row_non_mask]
    pf_dst_full = dst_full_all[row_non_mask]
    pf_src_non = src_non_all[row_non_mask]
    pf_dst_non = dst_non_all[row_non_mask]

    pf_dst_is_non = pf_dst_non >= 0

    g_diag = Gij[non_to_full, non_to_full]
    b_diag = Bij[non_to_full, non_to_full]

    g_val_pf = Gij[pf_src_full, pf_dst_full]
    b_val_pf = Bij[pf_src_full, pf_dst_full]
    return {
        "total_nodes": total_nodes,
        "non_slack": non_slack,
        "slack_idx": slack_idx,
        "non_to_full": non_to_full,
        "full_to_non": full_to_non,
        "pf_src_full": pf_src_full,
        "pf_dst_full": pf_dst_full,
        "pf_src_non": pf_src_non,
        "pf_dst_non": pf_dst_non,
        "pf_dst_is_non": pf_dst_is_non,
        "g_diag": g_diag,
        "b_diag": b_diag,
        "g_val_pf": g_val_pf,
        "b_val_pf": b_val_pf,
    }

def update_state_differentiable_sparse(state, out, Gij, Bij, slack_node, node_mean, node_std, output_mean, output_std, lines_mean, lines_std, pf_cache, update_edge_attr=True):
    device = state.x.device
    dtype = state.x.dtype

    num_graphs = state.num_graphs
    total_nodes = pf_cache["total_nodes"]
    non_slack = pf_cache["non_slack"]

    non_to_full = pf_cache["non_to_full"].to(device=device)
    slack_idx = pf_cache["slack_idx"]

    x_all = state.x.view(num_graphs, non_slack, -1)
    out_all = out.view(num_graphs, non_slack, -1)
    types_all = state.masks.view(num_graphs, non_slack)

    x_raw = x_all * node_std.view(1, 1, -1) + node_mean.view(1, 1, -1)
    out_phys = out_all * output_std.view(1, 1, -1) + output_mean.view(1, 1, -1)

    old_angle = x_raw[:, :, 4]
    old_voltage = x_raw[:, :, 5]

    delta_ang = out_phys[:, :, 0]
    delta_v = out_phys[:, :, 1]

    pq_mask = types_all == 1

    new_angle = old_angle + delta_ang
    new_voltage = torch.where(pq_mask, old_voltage + delta_v, old_voltage)

    p_spec = x_raw[:, :, 0] + x_raw[:, :, 2]
    q_spec = x_raw[:, :, 1] + x_raw[:, :, 3]

    slack_ang = slack_node[1].to(dtype=dtype, device=device)
    slack_vol = slack_node[2].to(dtype=dtype, device=device)

    angle_full = torch.zeros((num_graphs, total_nodes), device=device, dtype=dtype)
    voltage_full = torch.zeros((num_graphs, total_nodes), device=device, dtype=dtype)

    angle_full[:, non_to_full] = new_angle
    voltage_full[:, non_to_full] = new_voltage

    angle_full[:, slack_idx] = slack_ang
    voltage_full[:, slack_idx] = slack_vol

    # ------------------------------------------------------------
    #    P_ij = Vi Vj (Gij cos(theta) + Bij sin(theta))
    #    Q_ij = Vi Vj (Gij sin(theta) - Bij cos(theta))
    # ------------------------------------------------------------
    pf_src_full = pf_cache["pf_src_full"]
    pf_dst_full = pf_cache["pf_dst_full"]
    pf_src_non = pf_cache["pf_src_non"]

    g_val = pf_cache["g_val_pf"].to(dtype=dtype)
    b_val = pf_cache["b_val_pf"].to(dtype=dtype)

    theta_e = angle_full[:, pf_src_full] - angle_full[:, pf_dst_full]
    vi_e = voltage_full[:, pf_src_full]
    vj_e = voltage_full[:, pf_dst_full]

    cos_e = torch.cos(theta_e)
    sin_e = torch.sin(theta_e)

    adds_e = vi_e * vj_e * (g_val.view(1, -1) * cos_e + b_val.view(1, -1) * sin_e)
    mins_e = vi_e * vj_e * (g_val.view(1, -1) * sin_e - b_val.view(1, -1) * cos_e)

    right_hp = state.x.new_zeros((num_graphs, non_slack))
    right_hq = state.x.new_zeros((num_graphs, non_slack))

    src_index = pf_src_non.view(1, -1).expand(num_graphs, -1)
    right_hp.scatter_add_(dim=1, index=src_index, src=adds_e)
    right_hq.scatter_add_(dim=1, index=src_index, src=mins_e)

    delta_p = p_spec - right_hp
    delta_q = q_spec - right_hq
    delta_q = torch.where(pq_mask, delta_q, torch.zeros_like(delta_q))

    x_raw_new = torch.stack([right_hp, right_hq, delta_p, delta_q, new_angle, new_voltage], dim=-1,)

    x_norm_new = (x_raw_new - node_mean.view(1, 1, -1)) / node_std.view(1, 1, -1)
    x_norm_new = x_norm_new.reshape(-1, x_norm_new.shape[-1])

    new_state = copy(state)
    new_state.x = x_norm_new
    # Ablation: keep the initial Jacobian features fixed after state updates
    # return new_state
    if not update_edge_attr:
        # At the final correction of the final block, edge features have no
        # downstream consumer. Skipping their Jacobian rebuild preserves the
        # returned node state and its gradients while avoiding wasted work.
        return new_state

    g_diag = pf_cache["g_diag"].to(device=device, dtype=dtype)
    b_diag = pf_cache["b_diag"].to(device=device, dtype=dtype)

    self_add = new_voltage * new_voltage * g_diag.view(1, -1)
    self_min = -new_voltage * new_voltage * b_diag.view(1, -1)

    sum_add_excl_self = right_hp - self_add
    sum_min_excl_self = right_hq - self_min

    edge_index = state.edge_index
    src = edge_index[0]
    dst = edge_index[1]

    g_idx = state.batch[src]
    u_local = src % non_slack
    v_local = dst % non_slack

    u_full = non_to_full[u_local]
    v_full = non_to_full[v_local]

    g_edge = Gij[u_full, v_full].to(dtype=dtype)
    b_edge = Bij[u_full, v_full].to(dtype=dtype)

    theta_edge = new_angle[g_idx, u_local] - new_angle[g_idx, v_local]
    vu = new_voltage[g_idx, u_local]
    vv = new_voltage[g_idx, v_local]

    cos_edge = torch.cos(theta_edge)
    sin_edge = torch.sin(theta_edge)

    adds_edge = vu * vv * (g_edge * cos_edge + b_edge * sin_edge)
    mins_edge = vu * vv * (g_edge * sin_edge - b_edge * cos_edge)

    source_pq_edge = pq_mask[g_idx, u_local].to(dtype=dtype)
    target_pq_edge = pq_mask[g_idx, v_local].to(dtype=dtype)

    is_diag = u_local == v_local

    sum_add_excl_edge = sum_add_excl_self[g_idx, u_local]
    sum_min_excl_edge = sum_min_excl_self[g_idx, u_local]

    gii_edge = g_diag[u_local]
    bii_edge = b_diag[u_local]

    # H
    h_off = mins_edge
    h_diag = -sum_min_excl_edge
    h_edge = torch.where(is_diag, h_diag, h_off)

    # M
    m_off = (adds_edge / vv) * target_pq_edge
    m_diag = (sum_add_excl_edge / vu + 2.0 * gii_edge * vu) * source_pq_edge
    m_edge = torch.where(is_diag, m_diag, m_off)

    # K
    k_off = (-adds_edge) * source_pq_edge
    k_diag = sum_add_excl_edge * source_pq_edge
    k_edge = torch.where(is_diag, k_diag, k_off)

    # L
    l_off = (mins_edge / vv) * source_pq_edge * target_pq_edge
    l_diag = (sum_min_excl_edge / vu - 2.0 * bii_edge * vu) * source_pq_edge
    l_edge = torch.where(is_diag, l_diag, l_off)

    edge_raw_new = torch.stack([g_edge, b_edge, h_edge, m_edge, k_edge, l_edge], dim=1,)

    edge_norm_new = (edge_raw_new - lines_mean.view(1, -1)) / lines_std.view(1, -1)

    new_state.edge_attr = edge_norm_new

    return new_state

def run_unroll_stepwise_sparse(model, data, unroll_steps, Gij_GPU, Bij_GPU, slack_node_GPU, node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU, lines_mean_GPU, lines_std_GPU, pf_cache, update_final_edge_attr=True):
    if unroll_steps < 1:
        raise ValueError("UNROLL_STEPS must be >= 1")
    state = data
    step_records = []
    for step in range(unroll_steps):
        state_before = state
        out = model(state_before, step=step)
        state_after = update_state_differentiable_sparse(
            state_before, out, Gij_GPU, Bij_GPU, slack_node_GPU,
            node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU,
            lines_mean_GPU, lines_std_GPU, pf_cache,
            update_edge_attr=(update_final_edge_attr or step < unroll_steps - 1),
        )
        step_records.append({
            "step": step,
            "state_before_x": state_before.x,
            "out": out,
            "state_after_x": state_after.x,
        })
        state = state_after
    final_state = state
    return step_records, final_state

def log_print(message, log_file_path):
    print(message) 
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(str(message) + "\n") 
