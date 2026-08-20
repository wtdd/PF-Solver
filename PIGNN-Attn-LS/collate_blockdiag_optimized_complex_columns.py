from typing import Dict, List, Optional, Union
import torch

# --------------------------------------------------------------------------- #
#   Fields present in each per-grid sample produced by the Dataset class      #
# --------------------------------------------------------------------------- #
# Per-bus vectors (length N). Complex vectors are fine here.
_VECTOR_FIELDS = "bus_type U_start U_newton S_start S_newton".split()  # (N,)

# Two-channel per-bus tensors (mag, angle)
_TWO_CHANNEL_FIELDS = ("V_start", "V_newton")  # (N, 2)

# Per-edge tensors (variable length across samples). Includes complex Y_Lines and currents.
_EDGE_FIELDS = "Lines_connected Y_Lines Y_C_Lines".split()  # (E,...)

# Square matrix fields (N×N). Ybus is complex and may be None.
_MATRIX_FIELDS = ("Ybus",)

# --------------------------------------------------------------------------- #
#                               helpers                                        #
# --------------------------------------------------------------------------- #

def _concat_present(samples: List[Dict[str, torch.Tensor]], fields, dim: int = 0):
    """
    Concatenate only those fields that are present in ALL samples and are tensors.
    Skips fields missing in any sample or whose value is None.
    """
    out: Dict[str, torch.Tensor] = {}
    for f in fields:
        vals = [s.get(f, None) for s in samples]
        if all(isinstance(v, torch.Tensor) for v in vals):
            out[f] = torch.cat(vals, dim=dim)
    return out

# --------------------------------------------------------------------------- #
#                                main API                                      #
# --------------------------------------------------------------------------- #

def collate_blockdiag(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, Union[torch.Tensor, None]]:
    """Collate a list of per-grid samples into **one** block-diagonal mega-grid.

    Returns a dict with tensors batched to B=1. If any sample lacks Ybus,
    out['Ybus'] is set to None (downstream can build Y on-the-fly).
    """
    out: Dict[str, Union[torch.Tensor, None]] = {}

    # 1) concatenate per-bus / per-edge tensors that exist across all samples
    out.update(_concat_present(samples, _VECTOR_FIELDS, dim=0))       # ⇒ (M,)
    out.update(_concat_present(samples, _TWO_CHANNEL_FIELDS, dim=0))  # ⇒ (M, 2)
    out.update(_concat_present(samples, _EDGE_FIELDS, dim=0))         # ⇒ (E_total, ...)

    # 2) block-diagonalize Ybus only if EVERY sample has it (and it's a tensor)
    have_all_y = all(isinstance(s.get("Ybus", None), torch.Tensor) for s in samples)
    if have_all_y and len(samples) > 0:
        out["Ybus"] = torch.block_diag(*[s["Ybus"] for s in samples])  # ⇒ (M, M)
    else:
        out["Ybus"] = None  # downstream model will construct Y as needed

    # 3) bookkeeping: sizes/offsets
    device_for_meta = None
    if "bus_type" in out and isinstance(out["bus_type"], torch.Tensor):
        device_for_meta = out["bus_type"].device
    else:
        # fallback: pick the first tensor we find
        for s in samples:
            for v in s.values():
                if isinstance(v, torch.Tensor):
                    device_for_meta = v.device
                    break
            if device_for_meta is not None:
                break
        if device_for_meta is None:
            device_for_meta = torch.device("cpu")

    sizes = torch.tensor([s["bus_type"].numel() for s in samples], device=device_for_meta)
    offsets = torch.cat((sizes.new_zeros(1), torch.cumsum(sizes, 0)[:-1]))

    # 4) add explicit batch dimension (B = 1) for tensor values only
    for k, v in list(out.items()):
        if isinstance(v, torch.Tensor):
            out[k] = v.unsqueeze(0)  # ⇒ (1, ...)

    out["offsets"] = offsets               # (num_grids,)
    out["sizes"] = sizes                   # (num_grids,)

    return out