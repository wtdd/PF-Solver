# PIGNN-Attn-LS Baseline

This directory contains the physics-informed graph neural network baseline with edge-aware self-attention and Armijo line search. It predicts iterative voltage-angle and voltage-magnitude updates while using AC power-balance residuals as the default training objective.

## Files

| Path | Purpose |
| --- | --- |
| `main.py` | Command-line training and evaluation entry point. |
| `GNSMsg_SelfAttention_armijo.py` | Edge-aware self-attention model with iterative correction and Armijo search. |
| `GNSMsg_armijo.py` | Message-passing model without the self-attention block. |
| `pignn_attn_ls_data.py` | Common CSV loader, cache, losses, evaluation metrics, and result export. |
| `collate_blockdiag_optimized_complex_columns.py` | Optional block-diagonal batch collation utility. |
| `ckpt/` | IEEE-14/118 checkpoints, logs, in-distribution/extrapolation results, and N–R-selected numerical-divergence stress-test results. |

This baseline additionally requires `torch-scatter` with a build compatible with the installed PyTorch/CUDA version.

## Run

From the repository root, evaluate the included IEEE-118 checkpoint with:

```bash
python PIGNN-Attn-LS/main.py --data_dir Data/IEEE_118 --eval_only
```

Use `Data/IEEE_14` for IEEE-14. To train with the default attention model, omit `--eval_only`:

```bash
python PIGNN-Attn-LS/main.py --data_dir Data/IEEE_118
```

Important options include `--model`, `--K`, `--n_heads`, `--use_armijo`, `--batch_size`, `--lr`, and `--tol`. Use `--model GNSMsg` for the non-attention version and `python PIGNN-Attn-LS/main.py --help` for all settings.

Outputs are stored in `PIGNN-Attn-LS/ckpt/`: `*_PIGNN-Attn-LS_best.pt` is the best checkpoint, `.pt.log` records training and final evaluation, `<system>_test.log` is written in evaluation-only mode, `<system>.csv` contains cell-wise in-distribution/extrapolation results, and `<system>_ill.csv` contains N–R-selected numerical-divergence stress-case mismatches.
