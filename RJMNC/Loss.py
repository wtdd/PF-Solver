import torch
import torch.nn.functional as F

def _top_fraction_mean(values, fraction):
    if values.numel() == 0 or fraction <= 0.0:
        return values.new_tensor(0.0)
    flat = values.reshape(-1)
    k = max(1, int(round(float(flat.numel()) * float(fraction))))
    k = min(k, flat.numel())
    return torch.topk(flat, k=k, largest=True).values.mean()

def physics_loss_from_state_x_GPU_Vec(state_x, data, node_mean, node_std, cvar_quantile=0.75, tail_weight=0.0):
    num_graphs = data.num_graphs
    total_nodes = state_x.shape[0]
    non_slack = total_nodes // num_graphs

    x_raw = state_x.view(num_graphs, non_slack, -1) * node_std.view(1, 1, -1) + node_mean.view(1, 1, -1)

    # Columns 2 and 3 of x_raw_new from update_state_differentiable are delta_p and delta_q.
    delta_p = x_raw[:, :, 2]
    delta_q = x_raw[:, :, 3]

    types = data.masks.view(num_graphs, non_slack)
    pq_mask = (types == 1)
    pq_mask_f = pq_mask.to(dtype=state_x.dtype)

    p_err = delta_p.abs()
    q_err = delta_q.abs() * pq_mask_f

    count = non_slack + pq_mask_f.sum(dim=1)
    mean_residual_per_graph = (p_err.sum(dim=1) + q_err.sum(dim=1)) / count.clamp_min(1.0)
    mean_residual_loss = mean_residual_per_graph.mean()

    max_p_per_graph = p_err.max(dim=1).values
    max_q_per_graph = q_err.max(dim=1).values
    max_residual_per_graph = torch.maximum(max_p_per_graph, max_q_per_graph)

    max_residual_loss = max_residual_per_graph.mean()
    max_residual = max_residual_per_graph.max()

    tail_fraction = max(0.0, min(1.0, 1.0 - float(cvar_quantile)))
    tail_residual_loss = _top_fraction_mean(max_residual_per_graph, tail_fraction)

    physics_loss = mean_residual_loss + max_residual_loss + float(tail_weight) * tail_residual_loss
    return physics_loss, mean_residual_loss, max_residual_loss, max_residual


def physics_loss_from_state_steps_GPU_Vec(state_x_steps, data, node_mean, node_std):
    """Evaluate all unrolled states in one vectorized pass.

    This is mathematically identical to averaging
    ``physics_loss_from_state_x_GPU_Vec`` over the individual steps, but avoids
    launching the same small reduction kernels once per correction step.
    """
    if state_x_steps.dim() != 3 or state_x_steps.shape[0] < 1:
        raise ValueError("state_x_steps must have shape [steps, total_nodes, features].")

    num_steps, total_nodes, _ = state_x_steps.shape
    num_graphs = int(data.num_graphs)
    non_slack = total_nodes // num_graphs

    x_raw = (
        state_x_steps.view(num_steps, num_graphs, non_slack, -1)
        * node_std.view(1, 1, 1, -1)
        + node_mean.view(1, 1, 1, -1)
    )
    p_err = x_raw[..., 2].abs()
    q_err = x_raw[..., 3].abs()

    pq_mask = (data.masks.view(num_graphs, non_slack) == 1)
    pq_mask_f = pq_mask.to(dtype=state_x_steps.dtype).view(1, num_graphs, non_slack)
    q_err = q_err * pq_mask_f

    valid_count = (
        non_slack + pq_mask_f.sum(dim=2)
    ).clamp_min(1.0)
    mean_per_step = (
        (p_err.sum(dim=2) + q_err.sum(dim=2)) / valid_count
    ).mean(dim=1)
    max_per_graph = torch.maximum(p_err.amax(dim=2), q_err.amax(dim=2))
    max_per_step = max_per_graph.mean(dim=1)
    physics_per_step = mean_per_step + max_per_step

    return (
        physics_per_step.mean(),
        physics_per_step[-1],
        max_per_graph[-1].amax(),
    )

def stepwise_supervised_contract_loss_GPU_Vec_weight(step_records, data, node_mean, node_std, output_mean, output_std, shrink_ratio=0.8, eps_ang=None, eps_vol=None, eps_contract=1e-8, huber_delta=1.0, near_alpha=0.0, near_tau=1e-3):
    num_graphs = data.num_graphs
    total_nodes = data.x.shape[0]
    non_slack = total_nodes // num_graphs

    init_x = data.x.view(num_graphs, non_slack, -1)
    init_x_raw = init_x * node_std.view(1, 1, -1) + node_mean.view(1, 1, -1)

    y = data.y.view(num_graphs, non_slack, -1)
    y_raw = y * output_std.view(1, 1, -1) + output_mean.view(1, 1, -1)

    types = data.masks.view(num_graphs, non_slack)
    pq_mask = (types == 1)
    pq_mask_f = pq_mask.to(dtype=data.x.dtype)

    angle_true = init_x_raw[:, :, 4] + y_raw[:, :, 0]
    voltage_true = init_x_raw[:, :, 5].clone()
    voltage_true = torch.where(pq_mask, voltage_true + y_raw[:, :, 1], voltage_true)

    num_steps = len(step_records)
    if num_steps < 1:
        zero = data.x.new_tensor(0.0)
        return zero, zero, zero

    # Stack the short T dimension and perform each operation once.  No loss
    # term, weighting rule, or reduction is changed.
    state_before_x = torch.stack([
        record["state_before_x"]
        if "state_before_x" in record
        else record["state_before"].x
        for record in step_records
    ], dim=0)
    out_steps = torch.stack([record["out"] for record in step_records], dim=0)

    x_before_raw = (
        state_before_x.view(num_steps, num_graphs, non_slack, -1)
        * node_std.view(1, 1, 1, -1)
        + node_mean.view(1, 1, 1, -1)
    )
    out_phys = (
        out_steps.view(num_steps, num_graphs, non_slack, -1)
        * output_std.view(1, 1, 1, -1)
        + output_mean.view(1, 1, 1, -1)
    )

    pq_mask_s = pq_mask.view(1, num_graphs, non_slack)
    pq_mask_f_s = pq_mask_f.view(1, num_graphs, non_slack)
    remaining_ang = angle_true.unsqueeze(0) - x_before_raw[..., 4]
    remaining_vol = angle_true.new_zeros((num_steps, num_graphs, non_slack))
    remaining_vol = torch.where(
        pq_mask_s,
        voltage_true.unsqueeze(0) - x_before_raw[..., 5],
        remaining_vol,
    )
    pred_ang = out_phys[..., 0]
    pred_vol = torch.where(pq_mask_s, out_phys[..., 1], torch.zeros_like(out_phys[..., 1]))

    rel_err_ang = (pred_ang - remaining_ang) / eps_ang
    rel_err_vol = (pred_vol - remaining_vol) / eps_vol
    fit_ang = F.huber_loss(
        rel_err_ang, torch.zeros_like(rel_err_ang), reduction="none", delta=huber_delta
    )
    fit_vol = F.huber_loss(
        rel_err_vol, torch.zeros_like(rel_err_vol), reduction="none", delta=huber_delta
    ) * pq_mask_f_s

    valid_count = (non_slack + pq_mask_f.sum(dim=1)).clamp_min(1.0)
    fit_each_graph = (fit_ang + fit_vol).sum(dim=2) / valid_count.view(1, -1)

    e0_sq_graph = (
        remaining_ang.square() + remaining_vol.square() * pq_mask_f_s
    ).sum(dim=2)
    e1_sq_graph = (
        (remaining_ang - pred_ang).square()
        + (remaining_vol - pred_vol).square() * pq_mask_f_s
    ).sum(dim=2)

    if near_alpha > 0.0:
        near_weight = 1.0 + near_alpha * torch.exp(
            -e0_sq_graph.detach() / near_tau
        )
    else:
        near_weight = torch.ones_like(e0_sq_graph)

    fit_per_step = (
        near_weight * fit_each_graph
    ).sum(dim=1) / near_weight.sum(dim=1).clamp_min(1.0)

    eps_contract_active = 1e-6
    active_f = (e0_sq_graph > eps_contract_active).to(dtype=e1_sq_graph.dtype)
    contract_ratio = e1_sq_graph / e0_sq_graph.clamp_min(eps_contract)
    contract_each = F.relu(contract_ratio - shrink_ratio) * active_f
    contract_weight = near_weight * active_f
    contract_per_step = (
        contract_weight * contract_each
    ).sum(dim=1) / contract_weight.sum(dim=1).clamp_min(1.0)

    fit_loss = fit_per_step.mean()
    contract_loss = contract_per_step.mean()
    total_loss = fit_loss + contract_loss

    return total_loss, fit_loss, contract_loss
