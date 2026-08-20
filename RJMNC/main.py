from Function import *
from data import *
from Loss import *
from model import RJMNC
import os
import math
from datetime import datetime
import random
import time
import torch
import torch.nn.functional as F

# =============================================================================
# Basic runtime parameters
# This section only selects the system, model, and whether to enter test mode.
# =============================================================================
node_chose = "IEEE_118"
system_nodes = 117
message = ""
test_flag = False #True False

# =============================================================================
# Training parameters
# This section controls training speed, backpropagation, loss composition,
# gradient updates, and training-block sampling. Edit this section to change
# how the model learns.
# =============================================================================
EPOCHS, BATCH_SIZE= 2000, 512  # If CUDA OOM occurs with unrolling, reduce BATCH_SIZE (e.g., 512 or 256).
UNROLL_STEPS = 4
LEARNING_RATE = 0.001
SEED = 42
GRAD_CLIP_NORM = 5.0

# Supervised contraction-loss parameters
eps_contract, huber_delta = 1e-8, 1.0
shrink_ratio = 0.5   # The new error must be at most 50% of the previous error.
near_alpha = 3.0
near_tau = 2e-2

tol_mismatch =1E-4
inference_max_blocks = 20  # K in the paper; each block contains UNROLL_STEPS (= T) corrections.
gamma_1 = 1
gamma_2 = 0.2
gamma_3 = 3.0
gamma_4 = 0.35

e3_boost_first_two = 1.5
e3_target_ratio = 0.90
e3_focus_limit = 2e-2
e3_focus_smooth = 5e-3
e3_fine_weight = 1.0
e3_huber_beta = 2.0

# Multi-block training parameters
train_block_choices = (1, 2, 3)
train_block_probs = (0.10, 0.60, 0.30)
block_curriculum_warmup_epochs = 5
block_curriculum_transition_epochs = 40
block_probs_warmup = (0.75, 0.25, 0.0)
block_probs_target = train_block_probs
block_stage_weights = (1.0, 1.0, 0.75)
contract_boost_first_two = 1.5
truncate_between_blocks = True


# =============================================================================
# Validation parameters
# This section only controls the number of validation blocks, learning-rate
# scheduling, early stopping, and best-model selection. It does not directly
# affect training-loss backpropagation; edit it to change model selection.
# =============================================================================
val_max_blocks = 2
val_diag_max_blocks = 5

score_weight_conv1 = 1.0
score_weight_conv2 = 2.0
score_weight_conv3 = 1.0
score_weight_conv5 = 0.2
score_weight_res2 = 0.30
score_weight_tail2 = 0.02

SCHEDULER_PATIENCE  = 20
EARLY_STOP_PATIENCE = SCHEDULER_PATIENCE * 2
VAL_SCORE_IMPROVE_EPS = 1e-5

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_train_block_probs(epoch_idx):
    if not (len(train_block_choices) == len(block_probs_warmup) == len(block_probs_target)):
        raise ValueError("train_block_choices, block_probs_warmup, and block_probs_target must have the same length.")
    if epoch_idx < block_curriculum_warmup_epochs:
        return block_probs_warmup
    if epoch_idx < block_curriculum_transition_epochs:
        span = max(block_curriculum_transition_epochs - block_curriculum_warmup_epochs, 1)
        t = (epoch_idx - block_curriculum_warmup_epochs) / span
        return tuple(
            p0 + t * (p1 - p0)
            for p0, p1 in zip(block_probs_warmup, block_probs_target)
        )
    return block_probs_target

def sample_train_blocks(epoch_idx):
    block_probs = get_train_block_probs(epoch_idx)
    return random.choices(train_block_choices, weights=block_probs, k=1)[0]

def block_weight(block_idx):
    if block_idx < len(block_stage_weights):
        return block_stage_weights[block_idx]
    return block_stage_weights[-1]


def _first_existing_path(candidates, required_files, description):
    checked = []
    for candidate in candidates:
        if not candidate:
            continue
        path = os.path.abspath(candidate)
        checked.append(path)
        if all(os.path.exists(os.path.join(path, name)) for name in required_files):
            return path
    raise FileNotFoundError(
        f"Cannot find {description}. Checked:\n  " + "\n  ".join(checked)
    )


def resolve_data_path(system_name):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    configured_root = os.environ.get("RJMNC_DATA_ROOT")
    return _first_existing_path(
        [
            os.path.join(configured_root, system_name) if configured_root else None,
            os.path.join(project_dir, "..", "Data", system_name),
            os.path.join(project_dir, "..", "..", "服务器上的全量数据", "Python_Code", "Slover", "Data", system_name),
            os.path.join(project_dir, "..", "..", "Simulation", "Data", system_name),
        ],
        ("meta.csv", "bus_static.csv", "ybus.csv", "bus_state.csv", "jacobian_start.csv"),
        f"dataset for {system_name}",
    )


def resolve_test_checkpoint(checkpoint_name):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    configured_root = os.environ.get("RJMNC_CHECKPOINT_ROOT")
    filename = checkpoint_name + ".pt"
    checkpoint_dir = _first_existing_path(
        [
            configured_root,
            os.path.join(project_dir, "ckpt"),
            os.path.join(project_dir, "..", "..", "服务器上的全量数据", "Python_Code", "Slover", "PINN_slover", "ckpt"),
            os.path.join(project_dir, "..", "..", "Simulation", "PINN_slover", "ckpt"),
        ],
        (filename,),
        f"checkpoint {filename}",
    )
    return os.path.join(checkpoint_dir, filename)

def e3_excess_penalty_from_state_x(state_x, data, node_mean, node_std, tol, target_ratio=0.9, focus_limit=2e-2, focus_smooth=5e-3, fine_weight=1.0, huber_beta=2.0,):
    num_graphs = data.num_graphs
    total_nodes = state_x.shape[0]
    non_slack = total_nodes // num_graphs

    x_raw = state_x.view(num_graphs, non_slack, -1) * node_std.view(1, 1, -1) + node_mean.view(1, 1, -1)
    delta_p = x_raw[:, :, 2].abs()
    delta_q = x_raw[:, :, 3].abs()
    pq_mask = (data.masks.view(num_graphs, non_slack) == 1)
    q_max = torch.where(
        pq_mask.any(dim=1),
        torch.where(pq_mask, delta_q, torch.zeros_like(delta_q)).amax(dim=1),
        torch.zeros(num_graphs, device=state_x.device, dtype=state_x.dtype),
    )
    max_mismatch = torch.maximum(delta_p.amax(dim=1), q_max)
    ratio = max_mismatch / tol
    excess_ratio = F.relu(ratio - 1.0)
    log_penalty = torch.log1p(excess_ratio)

    # Extra pressure only near the E-3 band. This avoids dominating early
    # training while still separating 1.2e-3 from 9e-4 in the tail cases.
    fine_excess = F.relu(ratio - target_ratio)
    beta_t = state_x.new_tensor(huber_beta).clamp_min(1e-12)
    fine_penalty = torch.where(
        fine_excess < beta_t,
        0.5 * fine_excess.pow(2) / beta_t,
        fine_excess - 0.5 * beta_t,
    )
    focus = torch.sigmoid(
        (state_x.new_tensor(focus_limit) - max_mismatch.detach())
        / state_x.new_tensor(focus_smooth).clamp_min(1e-12)
    )
    return (log_penalty + fine_weight * focus * fine_penalty).mean()

def loss(model, data, Gij_GPU, Bij_GPU, slack_node_GPU, node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU, lines_mean_GPU, lines_std_GPU, eps_ang, eps_vol, pf_cache, unroll_step=UNROLL_STEPS, num_blocks=1):
    state = data
    weighted_loss_step = data.x.new_tensor(0.0)
    weighted_loss2_step = data.x.new_tensor(0.0)
    weighted_loss2_final = data.x.new_tensor(0.0)
    weighted_loss_e3 = data.x.new_tensor(0.0)
    abs_final_residual = data.x.new_tensor(0.0)
    total_weight = data.x.new_tensor(0.0)

    for block_idx in range(num_blocks):
        step_records, state = run_unroll_stepwise_sparse(
            model, state, unroll_step, Gij_GPU, Bij_GPU, slack_node_GPU,
            node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU,
            lines_mean_GPU, lines_std_GPU, pf_cache,
            update_final_edge_attr=(block_idx < num_blocks - 1),
        )

        loss_step_raw, loss_fit_step, loss_contract_step = stepwise_supervised_contract_loss_GPU_Vec_weight(step_records, data, node_mean_GPU, node_std_GPU, output_mean_GPU, output_std_GPU, shrink_ratio=shrink_ratio, eps_ang=eps_ang, eps_vol=eps_vol, eps_contract=eps_contract, huber_delta=huber_delta, near_alpha=near_alpha, near_tau=near_tau)

        if block_idx < 2:
            loss_step_eff = loss_fit_step + contract_boost_first_two * loss_contract_step
        else:
            loss_step_eff = loss_step_raw

        state_x_steps = torch.stack(
            [record["state_after_x"] for record in step_records], dim=0
        )
        loss2_step, loss2_final, abs_final_residual = (
            physics_loss_from_state_steps_GPU_Vec(
                state_x_steps, data, node_mean_GPU, node_std_GPU
            )
        )
        loss_e3 = e3_excess_penalty_from_state_x(state.x, data, node_mean_GPU, node_std_GPU, tol_mismatch, target_ratio=e3_target_ratio, focus_limit=e3_focus_limit, focus_smooth=e3_focus_smooth, fine_weight=e3_fine_weight, huber_beta=e3_huber_beta,)
        if block_idx < 2:
            loss_e3 = e3_boost_first_two * loss_e3

        w = data.x.new_tensor(block_weight(block_idx))
        total_weight = total_weight + w
        weighted_loss_step = weighted_loss_step + w * loss_step_eff
        weighted_loss2_step = weighted_loss2_step + w * loss2_step
        weighted_loss2_final = weighted_loss2_final + w * loss2_final
        weighted_loss_e3 = weighted_loss_e3 + w * loss_e3

        # Truncated BPTT across blocks: keep multi-block state distribution,
        # but avoid very expensive cross-block backprop graph growth.
        if truncate_between_blocks and block_idx < (num_blocks - 1):
            state.x = state.x.detach()
            if getattr(state, "edge_attr", None) is not None:
                state.edge_attr = state.edge_attr.detach()

    total_weight = total_weight.clamp_min(1e-12)
    return (
        weighted_loss_step / total_weight,
        weighted_loss2_step / total_weight,
        weighted_loss2_final / total_weight,
        weighted_loss_e3 / total_weight,
        abs_final_residual,
    )

def main():
    set_global_seed(SEED)
    Data_path = resolve_data_path(node_chose)
    file_name = f"./ckpt/{node_chose}.pt"
    os.makedirs(os.path.join(os.getcwd(), "ckpt"), exist_ok=True)
    if test_flag:
        Log_name = f"./ckpt/{node_chose}.pt" + '.test'
    else:
        Log_name = file_name +'.log'
    log_print(f"{message}", Log_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_path = os.path.join(os.getcwd(), file_name)
    Gij, Bij, slack_node = loadAdmittance(Data_path,device)
    pf_cache = build_pf_sparse_cache(Gij, Bij, slack_node)
    train_loader, val_loader, test_loader, ill_test_loader, node_mean, node_std, output_mean, output_std, line_mean, line_std = load_dataset_fast(Data_path, BATCH_SIZE, cache_dir=os.path.join(Data_path, "processed_cache"), force_rebuild=False, test_batch_size=1, device=device)
    model = RJMNC(num_nodes=system_nodes, max_steps=UNROLL_STEPS, hidden_channels=128, node_emb_dim=8, local_hidden_dim=64, gat2_channels=256, lin_hidden1=128, lin_hidden2=64).to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1E-5)
    scheduler = get_scheduler(optimizer, SCHEDULER_PATIENCE)
    eps_ang = float(output_std[0].item())
    eps_vol = float(output_std[1].item())
    count = 1
    log_print(
        f"Config -> Max_epochs:{EPOCHS}, lr:{LEARNING_RATE}, batch:{BATCH_SIZE}, "
        f"UNROLL_STEPS:{UNROLL_STEPS}, shrink_ratio:{shrink_ratio}, "
        f"gamma_1:{gamma_1}, gamma_2:{gamma_2}, gamma_3:{gamma_3}, "
        f"gamma_4:{gamma_4}, near_alpha:{near_alpha}, near_tau:{near_tau}, "
        f"train_blocks:{train_block_choices}, train_block_probs:{train_block_probs}, "
        f"curriculum_warmup:{block_curriculum_warmup_epochs}, curriculum_transition:{block_curriculum_transition_epochs}, "
        f"block_probs_warmup:{block_probs_warmup}, block_probs_target:{block_probs_target}, block_stage_weights:{block_stage_weights}, "
        f"contract_boost_first_two:{contract_boost_first_two}, e3_boost_first_two:{e3_boost_first_two}, "
        f"e3_target_ratio:{e3_target_ratio}, e3_focus_limit:{e3_focus_limit}, e3_focus_smooth:{e3_focus_smooth}, "
        f"e3_fine_weight:{e3_fine_weight}, e3_huber_beta:{e3_huber_beta}, "
        f"score_weights:(conv1={score_weight_conv1}, conv2={score_weight_conv2}, conv3={score_weight_conv3}, "
        f"conv5={score_weight_conv5}, excess2={score_weight_res2}, tail2={score_weight_tail2}), "
        f"seed:{SEED}, grad_clip_norm:{GRAD_CLIP_NORM}, val_score_eps:{VAL_SCORE_IMPROVE_EPS}, "
        f"truncate_between_blocks:{truncate_between_blocks}, "
        f"val_max_blocks:{val_max_blocks}, val_diag_max_blocks:{val_diag_max_blocks}, "
        f"inference_max_blocks:{inference_max_blocks}, tol_mismatch:{tol_mismatch}",
        Log_name
    )
    if test_flag:
        print("Start Testing")
        checkpoint_path = resolve_test_checkpoint(node_chose)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        test_batch_loader = DataLoader(test_loader.dataset, batch_size=512, shuffle=False, pin_memory=(device.type == "cuda"))
        test_unrolls_batch(model, test_batch_loader, device, Gij, Bij, slack_node, node_mean, node_std, output_mean, output_std, line_mean, line_std, unroll_steps=UNROLL_STEPS, max_blocks=inference_max_blocks, tol_mismatch=tol_mismatch, pf_cache=pf_cache, log_name=Log_name, node_chose=node_chose)
        ill_batch_loader = DataLoader(ill_test_loader.dataset, batch_size=100, shuffle=False, pin_memory=(device.type == "cuda"))
        ill_csv_path = os.path.join(os.path.dirname(__file__), "ckpt", f"{node_chose}_ill.csv")
        test_ill_conditioned_batch(model, ill_batch_loader, device, Gij, Bij, slack_node, node_mean, node_std, output_mean, output_std, line_mean, line_std, unroll_steps=UNROLL_STEPS, max_blocks=inference_max_blocks, tol_mismatch=tol_mismatch, pf_cache=pf_cache, log_name=Log_name, mismatch_csv_path=ill_csv_path)
        log_print(f"Testing File: {node_chose}.pt", Log_name)
        return
    log_print(f"Model:{file_name}", Log_name)
    best_score = float("-inf")
    best_key = (float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"), float("-inf"))
    best_residual2 = float("inf")
    bad_epochs = 0
    print("Begin Training")

    for epoch in range(EPOCHS):
        model.train()
        get_train_block_probs(epoch)

        total_loss, patch, count = None, 0, count + 1
        time0 = time.time()
        for data in train_loader:
            patch = patch + 1
            data = data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            num_blocks = sample_train_blocks(epoch)
            (loss_step_raw, loss2_step_raw, loss2_final_raw, loss_e3_raw, abs_final_residual) = loss(
                model, data, Gij, Bij, slack_node, node_mean, node_std, output_mean, output_std, line_mean, line_std,
                eps_ang, eps_vol, pf_cache, unroll_step=UNROLL_STEPS, num_blocks=num_blocks
            )
            loss1 = gamma_1 * loss_step_raw
            loss2 = gamma_2 * loss2_step_raw + gamma_3 * loss2_final_raw
            loss3 = gamma_4 * loss_e3_raw
            all_loss = (loss1 + loss2 + loss3)
            if patch % 25 == 0:
                (total_val, l1_step_val, l2_step_val, l2_final_val, l3_e3_val, final_residual_val) = torch.stack([
                    all_loss.detach(),
                    loss1.detach(),
                    (gamma_2 * loss2_step_raw).detach(),
                    (gamma_3 * loss2_final_raw).detach(),
                    loss3.detach(),
                    abs_final_residual.detach(),
                ]).cpu().tolist()
                total_for_ratio = total_val + 1e-12
                log_print(f"     {patch} total:{total_val:.5f} | blocks:{num_blocks} | L1_step:{l1_step_val:.5f} ({100 * l1_step_val / total_for_ratio:.1f}%) | L2_step:{l2_step_val:.5f} ({100 * l2_step_val / total_for_ratio:.1f}%) L2_final:{l2_final_val:.5f} ({100 * l2_final_val / total_for_ratio:.1f}%) | L3_e3:{l3_e3_val:.5f} ({100 * l3_e3_val / total_for_ratio:.1f}%) | final_max_PQ:{final_residual_val:.3e}", Log_name)
            detached_loss = all_loss.detach()
            total_loss = detached_loss if total_loss is None else total_loss + detached_loss
            all_loss.backward()
            if GRAD_CLIP_NORM is not None and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
        epoch_train_sec = time.time() - time0

        time0 = time.time()
        # Validation: prioritize low-block convergence and E-3 tail safety.
        val_metrics = eval_convergence_metrics_fast(model, val_loader, device, Gij, Bij, slack_node, node_mean, node_std, output_mean, output_std, line_mean, line_std, UNROLL_STEPS, pf_cache, tol_mismatch=tol_mismatch, max_blocks=val_diag_max_blocks)
        conv1 = val_metrics["conv_rates"][0]
        conv2 = val_metrics["conv_rates"][1] if len(val_metrics["conv_rates"]) > 1 else conv1
        conv3 = val_metrics["conv_rates"][2] if len(val_metrics["conv_rates"]) > 2 else conv2
        conv5 = val_metrics["conv_rates"][4] if len(val_metrics["conv_rates"]) > 4 else val_metrics["conv_rates"][-1]
        residual1 = val_metrics["residual_means"][0]
        residual2 = val_metrics["residual_means"][1] if len(val_metrics["residual_means"]) > 1 else val_metrics["residual_means"][0]
        residual3 = val_metrics["residual_means"][2] if len(val_metrics["residual_means"]) > 2 else residual2
        excess_means = val_metrics.get("excess_means", [])
        excess2 = excess_means[1] if len(excess_means) > 1 else max(0.0, residual2 / tol_mismatch - 1.0)
        residual_maxes = val_metrics.get("residual_maxes", [])
        residual2_max = residual_maxes[1] if len(residual_maxes) > 1 else residual2
        tail2_excess = max(0.0, residual2_max / tol_mismatch - 1.0)
        tail2_penalty = math.log1p(tail2_excess)
        val_score = (
            score_weight_conv1 * conv1
            + score_weight_conv2 * conv2
            + score_weight_conv3 * conv3
            + score_weight_conv5 * conv5
            - score_weight_res2 * excess2
            - score_weight_tail2 * tail2_penalty
        )
        current_key = (conv2, conv3, conv1, -excess2, -residual2, -residual2_max)
        epoch_val_sec = time.time() - time0
        print(f"Epoch Time(s): Train: {epoch_train_sec: .2f}, Val: {epoch_val_sec: .2f}")
        # Learning Rate
        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(1.0 - val_score) # Adjust the learning rate (mode=min).
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            log_print(f"🔻 LR reduced: {prev_lr:.2e} -> {current_lr:.2e}", Log_name)
        train_loss_avg = float((total_loss / patch).item())
        improved = False
        if val_score > best_score + VAL_SCORE_IMPROVE_EPS:
            improved = True
        elif abs(val_score - best_score) <= VAL_SCORE_IMPROVE_EPS:
            if residual2 < best_residual2 - 1e-8:
                improved = True
            elif abs(residual2 - best_residual2) <= 1e-8 and current_key > best_key:
                improved = True

        if improved:
            best_key = current_key
            best_score = val_score
            best_residual2 = residual2
            bad_epochs = 0
            torch.save(model.state_dict(), save_path)
            log_print(
                f"🟩 Epoch {epoch + 1} | Train loss = {train_loss_avg:.4f} | Val conv@1={conv1:.4f}, conv@2={conv2:.4f}, conv@3={conv3:.4f}, conv@5={conv5:.4f}, "
                f"residual@1={residual1:.6e}, residual@2={residual2:.6e}, residual@3={residual3:.6e}, excess@2={excess2:.4f}, tail@2={residual2_max:.6e}, score={val_score:.6f} | LR: {current_lr:.4e}", Log_name )
        else:
            bad_epochs += 1
            log_print(
                f"🔘 Epoch {epoch + 1} No Improvement ({bad_epochs}/{EARLY_STOP_PATIENCE}) | "
                f"Val conv@1={conv1:.4f}, conv@2={conv2:.4f}, conv@3={conv3:.4f}, conv@5={conv5:.4f}, "
                f"residual@1={residual1:.6e}, residual@2={residual2:.6e}",
                # f"excess@2={excess2:.4f}, tail@2={residual2_max:.6e}, score={val_score:.6f} | ",
                # f"Best conv@1={best_key[2]:.4f}, conv@2={best_key[0]:.4f}, conv@3={best_key[1]:.4f}, "
                # f"residual@2={best_residual2:.6e}, score={best_score:.6f}",
                Log_name
            )
            if bad_epochs >= EARLY_STOP_PATIENCE:
                log_print(f"🔘 Early stopping at epoch {epoch + 1}", Log_name)
                break
        # if hasattr(model, "gat_scale"):
        #     print(f"gat_scale: {model.gat_scale.item():.6e}")
    log_print("\nTesting model", Log_name)
    model.load_state_dict(torch.load(save_path, map_location=device))
    test_batch_loader = DataLoader(test_loader.dataset, batch_size=512, shuffle=False, pin_memory=(device.type == "cuda"))
    test_unrolls_batch(model, test_batch_loader, device, Gij, Bij, slack_node, node_mean, node_std, output_mean, output_std, line_mean, line_std, unroll_steps=UNROLL_STEPS, max_blocks=inference_max_blocks, tol_mismatch=tol_mismatch, pf_cache=pf_cache, log_name=Log_name, node_chose=node_chose)
    ill_batch_loader = DataLoader(ill_test_loader.dataset, batch_size=100, shuffle=False, pin_memory=(device.type == "cuda"))
    ill_csv_path = os.path.join(os.path.dirname(__file__), "ckpt", f"{node_chose}_ill.csv")
    test_ill_conditioned_batch(model, ill_batch_loader, device, Gij, Bij, slack_node, node_mean, node_std, output_mean, output_std, line_mean, line_std, unroll_steps=UNROLL_STEPS, max_blocks=inference_max_blocks, tol_mismatch=tol_mismatch, pf_cache=pf_cache, log_name=Log_name, mismatch_csv_path=ill_csv_path)

if __name__ == "__main__":
    main()
