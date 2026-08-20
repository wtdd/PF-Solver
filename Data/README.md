# Data

This directory contains the common CSV datasets used by RJMNC and the baseline methods. `IEEE_14/` and `IEEE_118/` have the same schema, allowing each method to use the same operating conditions, initial states, and reference power-flow solutions.

| File | Contents |
| --- | --- |
| `meta.csv` | Sample split, source, PF convergence/label validity, and ID/OOD perturbation ranges. |
| `bus_static.csv` | System-level bus types, limits, base values, and nominal injections. |
| `branch_static.csv` | Branch topology, impedance, charging, tap, and rating data. |
| `ybus.csv` | Real and imaginary entries of the bus-admittance matrix. |
| `bus_state.csv` | Per-sample starting state, specified injections, converged PF state, mismatches, and correction labels. |
| `jacobian_start.csv` | Per-sample Jacobian entries evaluated at the starting state. |
| `branch_flow.csv` | Per-sample start/reference branch power contributions exported with the common schema. |

Power-flow states, injections, mismatches, Jacobian entries, and branch-flow contributions consumed by the learning code are stored in per unit. Columns with explicit suffixes retain their stated units (for example, degrees, kV, MW, or Mvar), and `rateA`/`rateB`/`rateC` retain the rating convention of the source IEEE case. Bus and branch tables include both original identifiers and zero-based indices.

The committed files are reduced examples intended for format and code-interface inspection. Each bundled system contains 40 training records, 800 grid-test records, and 2 stress records. The checkpoints and logs were produced with the complete datasets (40,000 train/validation, 4,000 grid-test, and 100 stress records per system). Use the linked datasets for paper-result reproduction and [`Data_Generation/README.md`](../Data_Generation/README.md) for the construction procedure.

## Full datasets used in the paper

The complete datasets used for the paper experiments are available from Google Drive:

- [IEEE-14 dataset](https://drive.google.com/drive/folders/1xG1XVSixQiwosz85P3OnPWOuKR7KavLI?usp=drive_link)
- [IEEE-118 dataset](https://drive.google.com/drive/folders/1A3GP7j8FnvMiGl6qBQr67g3BpE1AY8Oy?usp=drive_link)
