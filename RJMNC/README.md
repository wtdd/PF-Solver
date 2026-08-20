# RJMNC

This directory contains the proposed **Residual-Guided Jacobian-Aware Multi-Step Neural Correction** solver for AC power flow. The same graph-attention network is reused over multiple correction phases. Its inputs combine power-flow state features, bus type, bus identity, phase conditioning, and admittance/Jacobian edge features; the state and Jacobian features are refreshed after each correction.

## Files

| Path | Purpose |
| --- | --- |
| `main.py` | Training and evaluation entry point; experiment settings are defined near the top. |
| `model.py` | Local and graph-attention branches, embeddings, phase conditioning, and correction head. |
| `data.py` | CSV loading, normalization, and PyTorch Geometric graph construction. |
| `Loss.py` | Supervised contraction and physics-residual losses. |
| `Function.py` | Differentiable PF updates, Jacobian refresh, evaluation, logging, and CSV export. |
| `ckpt/` | IEEE-14/118 checkpoints, logs, in-distribution/extrapolation results, and N–R-selected numerical-divergence stress-test results. |
| [`Ablation/`](Ablation/) | Single-component IEEE-118 ablations and their artifacts. |

## Data

The loader expects the following files under `../Data/<system>/`:

```text
meta.csv
bus_static.csv
ybus.csv
bus_state.csv
jacobian_start.csv
```

`branch_static.csv` supplies topology information to relevant baselines, while `branch_flow.csv` is an additional generator export; neither is required by RJMNC. If files are stored elsewhere, set `RJMNC_DATA_ROOT` to the directory containing the `IEEE_14/` and `IEEE_118/` folders, or set `RJMNC_CHECKPOINT_ROOT` to the directory containing the RJMNC `.pt` files.

## Run

Edit the basic runtime parameters near the top of `main.py`:

```python
node_chose = "IEEE_118"
system_nodes = 117
test_flag = False
```

Use `system_nodes = 13` for IEEE-14. `test_flag = False` trains a model and then evaluates the best checkpoint; `test_flag = True` loads `ckpt/<node_chose>.pt` and performs evaluation only.

```bash
cd RJMNC
python main.py
```

The principal settings are `EPOCHS`, `BATCH_SIZE`, `UNROLL_STEPS`, `LEARNING_RATE`, `tol_mismatch`, and `inference_max_blocks`. At inference, a sample stops when its maximum P/Q mismatch is no greater than `tol_mismatch`, or after the configured maximum number of correction blocks. If GPU memory is insufficient, reduce `BATCH_SIZE`.

## Outputs

Outputs are written to `ckpt/`:

- `IEEE_14.pt` and `IEEE_118.pt`: trained checkpoints;
- `*.pt.log`: training runs and their final evaluations;
- `*.pt.test`: evaluation-only logs created when `test_flag = True`;
- `IEEE_14.csv` and `IEEE_118.csv`: cell-wise in-distribution and operating-condition-extrapolation summaries;
- `*_ill.csv`: N–R-selected numerical-divergence stress-case mismatch results.

The full evaluation reports voltage-state errors against the reference PF solution, power mismatches, convergence rates, stopping reasons, and runtime. See the repository-level [`README.md`](../README.md) for the compact voltage-angle and voltage-magnitude error table.
