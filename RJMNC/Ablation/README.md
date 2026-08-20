# RJMNC Ablation Experiments

The ablations use the IEEE-118 dataset, training protocol, loss, stopping rule, and evaluation split of the full RJMNC model. Each variant is retrained from scratch with one component changed; all other settings remain unchanged. The relevant switches are left as comments in the source code.

| Variant | How it is performed |
| --- | --- |
| `Static Jacobian/` | In `Function.py`, enable the marked early return after the node-state update. Residual and state-dependent node features are still refreshed, while the initial Jacobian edge features are retained across corrections. |
| `Without Jacobian/` | Enable the marked zeroing of Jacobian columns `edge_attr[:, 2:6]` in `model.py` and the marked early return in `Function.py`, so no Jacobian information is supplied or refreshed. |
| `Without phase conditioning/` | Enable the marked zero tensor in `model.py` in place of the phase-conditioning embedding. The remaining input dimensions and model structure are unchanged. |

Each subdirectory contains the trained checkpoint, its `.pt.log` file, the cell-wise in-distribution/extrapolation result CSV, and the N–R-selected numerical-divergence stress-case CSV. These files are provided for direct result inspection; no separate source copy is required.
