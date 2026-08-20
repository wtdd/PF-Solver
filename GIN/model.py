import torch
from torch import nn

from data import PFSystemInfo, idx, mismatch_from_voltage, pack_state, polar_jacobian, unpack_state


def pack_mismatch(batch, vm, va, info: PFSystemInfo, ybus):
    dp, dq = mismatch_from_voltage(ybus, batch["p_spec"], batch["q_spec"], vm, va)
    device = vm.device
    return torch.cat([dp[:, idx(info.non_slack_idx, device)], dq[:, idx(info.pq_idx, device)]], dim=1)


class GlobalReceptiveGIN(nn.Module):
    """GIN Graph Iteration Network in reduced AC-PF coordinates.

    The paper writes the residual block as Eq. (15), the identity block as
    Eq. (16), and the final layer update as Eq. (17). Here the Newton-like
    correction is computed from the reduced PF Jacobian, then modulated by
    trainable Hadamard gains and biases, matching the non-activation iteration
    idea in Sec. III.
    """

    def __init__(self, state_dim: int, layers: int = 3):
        super().__init__()
        self.layers = layers
        self.gain = nn.Parameter(torch.ones(layers, state_dim))
        self.bias = nn.Parameter(torch.zeros(layers, state_dim))

    def forward(self, batch, info: PFSystemInfo, ybus: torch.Tensor):
        state = pack_state(batch["vm_start"], batch["va_start"], info)
        vm, va = unpack_state(state, batch, info)
        for layer in range(self.layers):
            mismatch = pack_mismatch(batch, vm, va, info, ybus)
            jac = polar_jacobian(ybus, vm, va, info)
            damp = torch.eye(jac.shape[-1], dtype=jac.dtype, device=jac.device).unsqueeze(0) * 1e-5
            # GIN Eq. (5): Newton correction from the inverse Jacobian.
            delta = torch.linalg.solve(jac + damp, mismatch.unsqueeze(-1)).squeeze(-1)
            # GIN Eq. (15)-(17): residual correction plus identity state.
            state = state + delta * self.gain[layer].unsqueeze(0) + self.bias[layer].unsqueeze(0)
            vm, va = unpack_state(state, batch, info)
        return state
