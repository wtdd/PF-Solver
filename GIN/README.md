# Global-Receptive Graph Iteration Network Baseline

This directory contains the GIN baseline adapted to the repository's common AC power-flow CSV format. The model operates in reduced PF coordinates and applies a sequence of Newton-like Jacobian corrections with trainable layer-wise gains and biases.

## Files

| Path | Purpose |
| --- | --- |
| `main.py` | Command-line training and evaluation entry point. |
| `model.py` | Global-Receptive Graph Iteration Network implementation. |
| `data.py` | CSV loader, PF system representation, Jacobian construction, and batching utilities. |
| `metrics.py` | Physics loss, voltage/mismatch metrics, ID/OOD summaries, N-R refinement, and CSV export. |
| `ckpt/` | IEEE-14/118 checkpoints, logs, in-distribution/extrapolation results, and N–R-selected numerical-divergence stress-test results. |

## Run

From the repository root, evaluate the included IEEE-118 checkpoint with:

```bash
python GIN/main.py --data_dir Data/IEEE_118
```

Use `Data/IEEE_14` for IEEE-14. Checkpoint loading is enabled by default when the matching file exists. To train a new model, run:

```bash
python GIN/main.py --data_dir Data/IEEE_118 --load_trained_model false
```

The main options include `--layers`, `--epochs`, `--batch_size`, `--lr`, and `--tol`; run `python GIN/main.py --help` for the complete list.

Outputs are stored in `GIN/ckpt/`: `*_GIN_best.pt` is the best checkpoint, `.pt.log` records training and final evaluation, `<system>_test.log` is written in checkpoint-only evaluation mode, `<system>.csv` contains cell-wise in-distribution/extrapolation results, and `<system>_ill.csv` contains N–R-selected numerical-divergence stress-case mismatches.
