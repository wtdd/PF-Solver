import torch
from torch import nn

from data import PFSystemInfo, calc_power


class TwoLayerMLP(nn.Module):
    """Paper Sec. 3.2: tanh hidden layer followed by a linear layer."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.tanh(self.linear1(x)))


class TypedGraphLayer(nn.Module):
    """One independently parameterized TGN layer from Eqs. (12)-(18)."""

    def __init__(self, hidden_dim: int, message_steps: int, decoder_init: str = "tiny"):
        super().__init__()
        d = hidden_dim
        self.message_steps = message_steps

        # Four node-type encoders gamma_i. Bus features have size 4; branch features have size 3.
        self.encode_pv = nn.Linear(4, d)
        self.encode_pq = nn.Linear(4, d)
        self.encode_slack = nn.Linear(4, d)
        self.encode_edge = nn.Linear(3, d)

        # Five directed message functions described in Sec. 3.2.
        self.pv_to_edge = TwoLayerMLP(d, d, d)
        self.pq_to_edge = TwoLayerMLP(d, d, d)
        self.slack_to_edge = TwoLayerMLP(d, d, d)
        self.edge_to_pv = TwoLayerMLP(d, d, d)
        self.edge_to_pq = TwoLayerMLP(d, d, d)

        # Slack states are fixed; PV, PQ and branch states are updated.
        self.update_pv = TwoLayerMLP(2 * d, d, d)
        self.update_pq = TwoLayerMLP(2 * d, d, d)
        self.update_edge = TwoLayerMLP(4 * d, d, d)

        self.decode_pv = nn.Linear(d, 1)
        self.decode_pq = nn.Linear(d, 2)
        if decoder_init == "tiny":
            # Stable adaptation: keep the initial 15-layer residual solver near flat start.
            nn.init.normal_(self.decode_pv.weight, mean=0.0, std=1e-3)
            nn.init.normal_(self.decode_pq.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.decode_pv.bias)
            nn.init.zeros_(self.decode_pq.bias)
        elif decoder_init == "glorot":
            # TensorFlow Dense uses Glorot by default; expose a decoder-init ablation.
            nn.init.xavier_uniform_(self.decode_pv.weight)
            nn.init.xavier_uniform_(self.decode_pq.weight)
            nn.init.zeros_(self.decode_pv.bias)
            nn.init.zeros_(self.decode_pq.bias)
        else:
            raise ValueError(f"Unknown decoder_init={decoder_init!r}; expected 'tiny' or 'glorot'")

    @staticmethod
    def _typed_bus_to_edge(
        messages: torch.Tensor,
        type_idx: torch.Tensor,
        n_bus: int,
        branch_from: torch.Tensor,
        branch_to: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, width = messages.shape
        full = messages.new_zeros((batch, n_bus, width))
        full = full.index_copy(1, type_idx, messages)
        return full.index_select(1, branch_from) + full.index_select(1, branch_to)

    @staticmethod
    def _edge_to_typed_bus(
        messages: torch.Tensor,
        type_idx: torch.Tensor,
        n_bus: int,
        branch_from: torch.Tensor,
        branch_to: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, width = messages.shape
        full = messages.new_zeros((batch, n_bus, width))
        full = full.index_add(1, branch_from, messages)
        full = full.index_add(1, branch_to, messages)
        return full.index_select(1, type_idx)

    def forward(
        self,
        x_pv: torch.Tensor,
        x_pq: torch.Tensor,
        x_slack: torch.Tensor,
        x_edge: torch.Tensor,
        pv_idx: torch.Tensor,
        pq_idx: torch.Tensor,
        slack_idx: torch.Tensor,
        branch_from: torch.Tensor,
        branch_to: torch.Tensor,
        n_bus: int,
    ):
        z_pv = self.encode_pv(x_pv)
        z_pq = self.encode_pq(x_pq)
        z_slack = self.encode_slack(x_slack)
        z_edge = self.encode_edge(x_edge)

        # The same mu and phi weights are reused for all L message/update steps.
        for _ in range(self.message_steps):
            pv_edge = self._typed_bus_to_edge(
                self.pv_to_edge(z_pv), pv_idx, n_bus, branch_from, branch_to
            )
            pq_edge = self._typed_bus_to_edge(
                self.pq_to_edge(z_pq), pq_idx, n_bus, branch_from, branch_to
            )
            slack_edge = self._typed_bus_to_edge(
                self.slack_to_edge(z_slack), slack_idx, n_bus, branch_from, branch_to
            )
            edge_pv = self._edge_to_typed_bus(
                self.edge_to_pv(z_edge), pv_idx, n_bus, branch_from, branch_to
            )
            edge_pq = self._edge_to_typed_bus(
                self.edge_to_pq(z_edge), pq_idx, n_bus, branch_from, branch_to
            )

            next_edge = self.update_edge(torch.cat([z_edge, pv_edge, pq_edge, slack_edge], dim=-1))
            next_pv = self.update_pv(torch.cat([z_pv, edge_pv], dim=-1))
            next_pq = self.update_pq(torch.cat([z_pq, edge_pq], dim=-1))
            z_edge, z_pv, z_pq = next_edge, next_pv, next_pq

        delta_theta_pv = self.decode_pv(z_pv).squeeze(-1)
        pq_output = self.decode_pq(z_pq)
        delta_vm_pq = pq_output[..., 0]
        delta_theta_pq = pq_output[..., 1]
        return delta_theta_pv, delta_vm_pq, delta_theta_pq


class TypedGraphPowerFlow(nn.Module):
    """Physics-informed typed graph power-flow solver from Lopez-Garcia et al.

    T TGN layers are independently parameterized. Each layer recomputes the AC
    power balance and applies L shared message/update steps on Slack, PV, PQ and
    branch node types before predicting a residual voltage update.
    """

    def __init__(
        self,
        info: PFSystemInfo,
        tgn_layers: int = 15,
        message_steps: int = 2,
        hidden_dim: int = 16,
        decoder_init: str = "tiny",
    ):
        super().__init__()
        if tgn_layers <= 0 or message_steps <= 0 or hidden_dim <= 0:
            raise ValueError("tgn_layers, message_steps and hidden_dim must all be positive")
        self.n_bus = info.n_bus
        self.tgn_layers = tgn_layers
        self.message_steps = message_steps
        self.hidden_dim = hidden_dim
        self.decoder_init = decoder_init
        self.layers = nn.ModuleList(
            [TypedGraphLayer(hidden_dim, message_steps, decoder_init=decoder_init) for _ in range(tgn_layers)]
        )
        self.register_buffer("pv_idx", torch.tensor(info.pv_idx, dtype=torch.long))
        self.register_buffer("pq_idx", torch.tensor(info.pq_idx, dtype=torch.long))
        self.register_buffer("non_slack_idx", torch.tensor(info.non_slack_idx, dtype=torch.long))
        self.register_buffer("slack_idx", torch.tensor([info.slack_idx], dtype=torch.long))
        self.register_buffer("branch_from", torch.tensor(info.branch_from, dtype=torch.long))
        self.register_buffer("branch_to", torch.tensor(info.branch_to, dtype=torch.long))
        self.register_buffer("branch_features", torch.tensor(info.branch_features, dtype=torch.float32))

    def _flat_start(self, batch):
        batch_size = batch["p_spec"].shape[0]
        vm = torch.ones_like(batch["p_spec"])
        slack_vm = batch["vm_start"].index_select(1, self.slack_idx)
        vm = vm.index_copy(1, self.slack_idx, slack_vm)
        if self.pv_idx.numel():
            vm = vm.index_copy(1, self.pv_idx, batch["vm_start"].index_select(1, self.pv_idx))
        slack_angle = batch["va_start"].index_select(1, self.slack_idx)
        va = slack_angle.expand(batch_size, self.n_bus).clone()
        return vm, va

    def forward_voltage(self, batch, ybus: torch.Tensor):
        vm, va = self._flat_start(batch)
        batch_size = vm.shape[0]
        edge_features = self.branch_features.to(vm.dtype).unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            p_calc, q_calc = calc_power(ybus, vm, va)
            dp = batch["p_spec"] - p_calc
            dq = batch["q_spec"] - q_calc

            # Eq. (12). For PV buses q_spec contains the fixed net demand part;
            # q_calc - q_spec is therefore the locally compensated Q generation.
            x_pv = torch.stack(
                [
                    vm.index_select(1, self.pv_idx),
                    va.index_select(1, self.pv_idx),
                    dp.index_select(1, self.pv_idx),
                    (q_calc - batch["q_spec"]).index_select(1, self.pv_idx),
                ],
                dim=-1,
            )
            x_pq = torch.stack(
                [
                    vm.index_select(1, self.pq_idx),
                    va.index_select(1, self.pq_idx),
                    dp.index_select(1, self.pq_idx),
                    dq.index_select(1, self.pq_idx),
                ],
                dim=-1,
            )
            x_slack = torch.stack(
                [
                    vm.index_select(1, self.slack_idx),
                    va.index_select(1, self.slack_idx),
                    p_calc.index_select(1, self.slack_idx),
                    q_calc.index_select(1, self.slack_idx),
                ],
                dim=-1,
            )

            dtheta_pv, dvm_pq, dtheta_pq = layer(
                x_pv,
                x_pq,
                x_slack,
                edge_features,
                self.pv_idx,
                self.pq_idx,
                self.slack_idx,
                self.branch_from,
                self.branch_to,
                self.n_bus,
            )
            dtheta = torch.zeros_like(va)
            dtheta = dtheta.index_copy(1, self.pv_idx, dtheta_pv)
            dtheta = dtheta.index_copy(1, self.pq_idx, dtheta_pq)
            dvm = torch.zeros_like(vm).index_copy(1, self.pq_idx, dvm_pq)
            va = va + dtheta
            # Paper Eq. (17): unconstrained residual voltage-magnitude update.
            vm = vm + dvm

        return vm, va

    def forward(self, batch, info: PFSystemInfo, ybus: torch.Tensor):
        del info  # Static graph metadata is registered in the model at construction.
        vm, va = self.forward_voltage(batch, ybus)
        return torch.cat(
            [va.index_select(1, self.non_slack_idx), vm.index_select(1, self.pq_idx)],
            dim=1,
        )
