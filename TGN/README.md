# Typed Graph Network Baseline

This directory contains the physics-informed Typed Graph Network (TGN) baseline adapted to the common CSV power-flow dataset. The default model uses 15 typed graph layers, two message/update steps per layer, and a hidden dimension of 16.

From the repository root, evaluate the included IEEE-118 checkpoint with:

```bash
python TGN/main.py --data_dir Data/IEEE_118
```

Use `Data/IEEE_14` for IEEE-14. Checkpoint loading is enabled by default when a compatible checkpoint exists. To train instead, pass `--load_trained_model false`; use `python TGN/main.py --help` for the remaining options.

Checkpoints, logs, cell-wise in-distribution/extrapolation CSV files, and N–R-selected numerical-divergence stress-test CSV files are stored in `TGN/ckpt/`. Training writes a `.pt.log`, while compatible-checkpoint evaluation writes `<system>_test.log`. Training minimizes the AC active/reactive power-balance residual; reference PF voltages are used for evaluation metrics only.
