# GNSMsg_EdgeSelfAttn.py (optimized)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max

# --------------------------- utils ---------------------------

def _segmented_softmax(logits_b_e_h: torch.Tensor, dst_e: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Vectorized softmax over incoming edges per (batch, head, node).
    logits_b_e_h: (B, E, H)
    dst_e:        (E,) destination node index in [0..N)
    returns:      (B, E, H) normalized weights
    """
    B, E, H = logits_b_e_h.shape
    device = logits_b_e_h.device
    N = num_nodes
    # Build a single segment-id that encodes (batch, head, dst)
    b_ids = torch.arange(B, device=device).view(B, 1, 1)
    h_ids = torch.arange(H, device=device).view(1, 1, H)
    dst = dst_e.view(1, E, 1)
    seg = (b_ids * (N * H)) + (h_ids * N) + dst              # (B,E,H)
    seg = seg.reshape(-1)                                    # (B*E*H,)

    src = logits_b_e_h.reshape(-1)                           # (B*E*H,)

    # subtract segment-wise max for stability
    max_per_seg, _ = scatter_max(src, seg, dim=0, dim_size=B * N * H)
    max_g = max_per_seg.index_select(0, seg)                 # gather
    x = torch.exp(src - max_g)
    denom = scatter_add(x, seg, dim=0, dim_size=B * N * H)
    denom_g = denom.index_select(0, seg)
    alpha = (x / (denom_g + 1e-12)).reshape(B, E, H)         # (B,E,H)
    return alpha

def _batched_mismatch_inf_norm(Y, v, th, P_set, Q_set, slack_mask, pv_mask):
    """
    Compute ∞-norm of power mismatch for (possibly) batched candidate voltages.
    Y:      (B?, N, N) or (1, N, N)
    v, th:  (..., N)
    Returns: scalar tensor
    """
    # Broadcast Y across leading alpha/batch dims
    # Vc: (..., N) complex
    Vc = v * torch.exp(1j * th)
    # align Y to match ... dims
    Yb = Y
    while Yb.dim() < Vc.dim() + 1:  # want Yb.shape == (..., N, N)
        Yb = Yb.unsqueeze(0)
    Ic = torch.matmul(Yb, Vc.unsqueeze(-1)).squeeze(-1)
    Sc = Vc * Ic.conj()
    DP = (P_set - Sc.real)
    DQ = (Q_set - Sc.imag)
    # broadcast masks
    DP = DP.masked_fill(slack_mask, 0.0)
    DQ = DQ.masked_fill(slack_mask | pv_mask, 0.0)
    DP_max = DP.abs().amax(dim=-1)
    DQ_max = DQ.abs().amax(dim=-1)
    # Keep every leading dimension (candidate and/or batch).  Armijo must pick
    # a step independently for each graph; reducing across the batch makes one
    # difficult sample stall all other samples and makes results batch-size
    # dependent.
    return torch.maximum(DP_max, DQ_max)

# --------------------- attention block -----------------------

class EdgeSelfAttnBlock(nn.Module):
    """
    Sparse graph self-attention with edge bias.
    Uses segmented-softmax (vectorized) over incoming edges per (batch, head, node).
    """
    def __init__(self, d_model: int, n_heads: int, edge_feat_dim: int, ffn_hidden: int = None, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.h = n_heads
        self.dh = d_model // n_heads

        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.edge_bias = nn.Sequential(
            nn.Linear(edge_feat_dim, max(8, edge_feat_dim * 2)),
            nn.LeakyReLU(0.1),
            nn.Linear(max(8, edge_feat_dim * 2), self.h)  # per-head bias
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        hid = ffn_hidden or (4 * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Linear(hid, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index_dir: torch.Tensor, edge_feat_dir: torch.Tensor):
        """
        x:             (B, N, D)
        edge_index_dir:(E, 2) directed (src=j, dst=i)
        edge_feat_dir: (E, F)
        """
        B, N, D = x.shape
        device = x.device
        src = edge_index_dir[:, 0]
        dst = edge_index_dir[:, 1]

        y = self.ln1(x)
        Q = self.q(y).view(B, N, self.h, self.dh)   # (B,N,H,dh)
        K = self.k(y).view(B, N, self.h, self.dh)
        V = self.v(y).view(B, N, self.h, self.dh)

        Qi = Q[:, dst, :, :]                        # (B,E,H,dh)
        Kj = K[:, src, :, :]                        # (B,E,H,dh)
        Vj = V[:, src, :, :]                        # (B,E,H,dh)

        logits = (Qi * Kj).sum(dim=-1) / math.sqrt(self.dh)    # (B,E,H)
        bias = self.edge_bias(edge_feat_dir).unsqueeze(0)      # (1,E,H)
        logits = logits + bias

        # segmented softmax over incoming edges per (batch, head, node)
        alpha = _segmented_softmax(logits, dst, N)             # (B,E,H)
        attn_msg = alpha.unsqueeze(-1) * Vj                    # (B,E,H,dh)

        # aggregate to dst nodes
        out = torch.zeros(B, N, self.h, self.dh, device=device, dtype=x.dtype)
        out.index_add_(1, dst, attn_msg)                       # (B,N,H,dh)
        out = self.drop(self.out(out.reshape(B, N, D)))
        x = x + out

        z = self.ln2(x)
        z = self.drop(self.ffn(z))
        return x + z

# ------------------ main model ------------------

class GNSMsg_EdgeSelfAttn(nn.Module):
    def __init__(
        self,
        d: int = 10,
        d_hi: int = 32,
        K: int = 30,
        pinn: bool = True,
        gamma: float = 0.9,
        v_limit: bool = True,
        use_armijo: bool = True,
        d_model: int = None,
        n_heads: int = 4,
        num_attn_layers: int = 1,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        self.K, self.d, self.d_hi = K, d, d_hi
        self.pinn, self.gamma, self.v_limit, self.use_armijo = pinn, gamma, v_limit, use_armijo

        self.d_model = d_model if d_model is not None else d_hi
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.num_attn_layers = num_attn_layers

        self.bus_feat_dim = 4 + d      # [v, θ, ΔP, ΔQ] + m
        self.edge_feat_dim = 3         # [Ysr, Ysi, Yc]

        self.in_proj = nn.Linear(self.bus_feat_dim, self.d_model)
        self.blocks = nn.ModuleList([
            EdgeSelfAttnBlock(self.d_model, self.n_heads, self.edge_feat_dim,
                              ffn_hidden=4 * self.d_model, dropout=attn_dropout)
            for _ in range(self.num_attn_layers)
        ])

        # per-iteration heads
        self.theta_head = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.v_head     = nn.ModuleList([nn.Linear(self.d_model, 1) for _ in range(K)])
        self.m_head     = nn.ModuleList([nn.Linear(self.d_model, d) for _ in range(K)])

        for k in range(K):
            nn.init.zeros_(self.theta_head[k].weight); nn.init.zeros_(self.theta_head[k].bias)
            nn.init.zeros_(self.v_head[k].weight);     nn.init.zeros_(self.v_head[k].bias)
            nn.init.zeros_(self.m_head[k].weight);     nn.init.zeros_(self.m_head[k].bias)

        # cache for undirected pair indices per graph size
        self._pair_cache = {}  # n -> (pairs_n on device)

    @torch.no_grad()
    def _pairs_for_n(self, n: int, device: torch.device) -> torch.Tensor:
        """
        Return upper-triangular undirected pairs for n nodes: shape (n*(n-1)//2, 2)
        Cached per (n, device).
        """
        key = (n, device)
        if key in self._pair_cache:
            return self._pair_cache[key]
        # vectorized triu
        iu = torch.triu_indices(n, n, offset=1, device=device)
        pairs = iu.t().contiguous()  # (e_all, 2)
        self._pair_cache[key] = pairs
        return pairs

    def forward(self, bus_type, Line, Y, Ys, Yc, S, V0, n_nodes_per_graph):
        """
        If Y is None, construct the dense admittance matrix from upper-tri edges (Line) and per-line Ys, Yc.
        Otherwise, use the provided Y. Works for block-diag batching (B==1, N=sum(subgraphs)) and plain batching.
        """
        device = bus_type.device
        B, N = bus_type.shape

        P_set, Q_set = S.real, S.imag
        v = V0[..., 0].clone()
        th = V0[..., 1].clone()
        m = torch.zeros(B, N, self.d, device=device)

        # ---- Helper: fast dense Y assembly from undirected edges ----
        def _build_dense_Y(N_local, edges_undirected, ys_edge, yc_edge):
            """
            edges_undirected: (E,2) long (global node indices)
            ys_edge: (E,) complex (series admittances)
            yc_edge: (E,) complex/real (line charging contribution per bus)
            """
            if edges_undirected.numel() == 0:
                # No lines -> zero matrix (no separate shunt provided).
                dtype = ys_edge.dtype if ys_edge.numel() else torch.complex64
                return torch.zeros(N_local, N_local, dtype=dtype, device=device)

            i = edges_undirected[:, 0]
            j = edges_undirected[:, 1]

            if not torch.is_complex(ys_edge):
                ys_edge = ys_edge.to(torch.complex64)
            yc_edge = yc_edge.to(ys_edge.dtype)

            Yloc = torch.zeros(N_local, N_local, dtype=ys_edge.dtype, device=device)

            # Off-diagonals: Y_ij = Y_ji = -Ys
            Yloc.index_put_((i, j), -ys_edge, accumulate=True)
            Yloc.index_put_((j, i), -ys_edge, accumulate=True)

            # Diagonals: sum incident series + charging
            diag = torch.zeros(N_local, dtype=ys_edge.dtype, device=device)
            diag.index_add_(0, i, ys_edge)
            diag.index_add_(0, j, ys_edge)
            diag.index_add_(0, i, yc_edge)
            diag.index_add_(0, j, yc_edge)

            Yloc.diagonal().add_(diag)
            return Yloc

        # -------- Build edge lists (once) and optionally Y --------
        shared_topology = n_nodes_per_graph is None and Line.dim() == 1
        if n_nodes_per_graph is not None:
            # Block-diag batching (recommended): B==1; treat lines as 1D concatenated per-graph
            Line = Line.squeeze(0) if Line.dim() == 2 else Line
            Ys = Ys.squeeze(0)
            Yc = Yc.squeeze(0)

            Ysr, Ysi = Ys.real, Ys.imag

            edge_index_parts, edge_feat_parts = [], []
            ys_parts, yc_parts = [], []
            ptr = 0
            offset = 0
            for n in n_nodes_per_graph:
                n = int(n)
                e_all = n * (n - 1) // 2
                mask_g = Line[ptr:ptr + e_all].bool()
                if mask_g.any():
                    pairs_g = self._pairs_for_n(n, device)  # (e_all, 2) local pairs
                    e_idx_g = pairs_g[mask_g] + offset  # globalize
                    edge_index_parts.append(e_idx_g)

                    # edge features for attention blocks
                    feat_g = torch.stack([Ysr[ptr:ptr + e_all][mask_g],
                                          Ysi[ptr:ptr + e_all][mask_g],
                                          Yc[ptr:ptr + e_all][mask_g].to(Ysr.dtype)], dim=-1)
                    edge_feat_parts.append(feat_g)

                    ys_parts.append(Ys[ptr:ptr + e_all][mask_g])
                    yc_parts.append(Yc[ptr:ptr + e_all][mask_g])
                ptr += e_all
                offset += n

            if edge_index_parts:
                undirected = torch.cat(edge_index_parts, dim=0)  # (E,2)
                edge_feat = torch.cat(edge_feat_parts, dim=0)  # (E,3)
                ys_edge = torch.cat(ys_parts, dim=0)  # (E,)
                yc_edge = torch.cat(yc_parts, dim=0)  # (E,)
            else:
                undirected = torch.empty(0, 2, dtype=torch.long, device=device)
                edge_feat = torch.empty(0, 3, dtype=Ysr.dtype, device=device)
                ys_edge = torch.empty(0, dtype=Ys.dtype, device=device)
                yc_edge = torch.empty(0, dtype=Ys.real.dtype, device=device)

            # directed duplication for attention
            edge_index_dir = torch.cat([undirected, undirected[:, [1, 0]]], dim=0)  # (2E,2)
            edge_feat_dir = torch.cat([edge_feat, edge_feat], dim=0)  # (2E,3)

            # Build Y only if needed
            if Y is None:
                Y = _build_dense_Y(N, undirected, ys_edge, yc_edge)  # (N,N) for B==1

        elif shared_topology:
            # Slover's IEEE data vary operating points but share one topology.
            # Build edges once and run attention over the whole true batch.
            pairs = self._pairs_for_n(N, device)
            mask = Line.bool()
            undirected = pairs[mask]
            ysr, ysi = Ys.real, Ys.imag
            edge_feat = torch.stack(
                [ysr[mask], ysi[mask], Yc[mask].to(ysr.dtype)], dim=-1
            )
            edge_index_dir = torch.cat([undirected, undirected[:, [1, 0]]], dim=0)
            edge_feat_dir = torch.cat([edge_feat, edge_feat], dim=0)
            if Y is None:
                Y = _build_dense_Y(N, undirected, Ys[mask], Yc[mask])
        else:
            # Plain batching fallback (all graphs share N)
            pairs = self._pairs_for_n(N, device)
            edge_index_dir_list, edge_feat_dir_list = [], []
            Y_list = [] if Y is None else None

            for b in range(B):
                mask = Line[b].bool()
                e_b = pairs[mask]
                Ysr_b, Ysi_b, Yc_b = Ys[b].real, Ys[b].imag, Yc[b]

                if e_b.numel() > 0:
                    feat_b = torch.stack([Ysr_b[mask], Ysi_b[mask], Yc_b[mask].to(Ysr_b.dtype)], dim=-1)
                    edge_index_dir_list.append(torch.cat([e_b, e_b[:, [1, 0]]], dim=0))
                    edge_feat_dir_list.append(torch.cat([feat_b, feat_b], dim=0))
                else:
                    edge_index_dir_list.append(torch.empty(0, 2, dtype=torch.long, device=device))
                    edge_feat_dir_list.append(torch.empty(0, 3, dtype=Ysr_b.dtype, device=device))

                if Y is None:
                    ys_edge_b = Ys[b][mask]
                    yc_edge_b = Yc[b][mask]
                    Y_list.append(_build_dense_Y(N, e_b, ys_edge_b, yc_edge_b))

            if Y is None:
                Y = torch.stack(Y_list, dim=0)  # (B,N,N)

        # Masks
        slack_mask = (bus_type == 1)
        pv_mask = (bus_type == 2)

        phys_terms = []  # collect per-iteration physics terms

        # ------------------------- K iterations -------------------------
        for k in range(self.K):
            # power mismatches
            Vc = v * torch.exp(1j * th)
            Ic = torch.matmul(Y, Vc.unsqueeze(-1)).squeeze(-1)  # supports (N,N) or (B,N,N)
            Sc = Vc * Ic.conj()
            DP = (P_set - Sc.real)
            DQ = (Q_set - Sc.imag)
            DP = DP.masked_fill(slack_mask, 0.0)
            DQ = DQ.masked_fill(slack_mask | pv_mask, 0.0)

            bus_feat = torch.stack([v, th, DP, DQ], dim=-1)  # (B,N,4)
            ctx = self.in_proj(torch.cat([bus_feat, m], dim=-1))

            # GNN message passing
            if n_nodes_per_graph is not None or shared_topology:
                x = ctx
                for blk in self.blocks:
                    x = blk(x, edge_index_dir, edge_feat_dir)  # (B,N,D)
            else:
                x_parts = []
                for b in range(B):
                    xb = ctx[b:b + 1]
                    if edge_index_dir_list[b].numel() == 0:
                        x_parts.append(xb)
                        continue
                    e_b, ef_b = edge_index_dir_list[b], edge_feat_dir_list[b]
                    for blk in self.blocks:
                        xb = blk(xb, e_b, ef_b)
                    x_parts.append(xb)
                x = torch.cat(x_parts, dim=0)

            dth = self.theta_head[k](x).squeeze(-1)
            dv = self.v_head[k](x).squeeze(-1)
            dm = torch.tanh(self.m_head[k](x))
            dm = F.layer_norm(dm, dm.shape[-1:])

            # constraints
            dth = dth.clone();
            dv = dv.clone()
            dth = dth.masked_fill(slack_mask, 0.0)
            dv = dv.masked_fill(slack_mask | pv_mask, 0.0)

            if self.v_limit:
                dtheta_max = 0.30
                dvm_frac = 0.10
                v_abs = v.abs()
                dth = torch.clamp(dth, -dtheta_max, dtheta_max)
                dv = torch.clamp(dv, -dvm_frac * v_abs, dvm_frac * v_abs)

            # ---- Armijo line search (optional) ----
            if self.use_armijo:
                v_min, v_max = 0.8, 1.2

                # Decision-only: evaluate current mismatch without autograd.
                # DP/DQ are the already-computed current residuals.  Reusing
                # them avoids one extra complex Y@V per correction step.
                with torch.no_grad():
                    F0 = torch.maximum(DP.abs().amax(dim=-1), DQ.abs().amax(dim=-1))

                alphas = v.new_tensor([1.0, 0.5, 0.25, 0.125, 0.0625])
                T = int(alphas.numel())

                # Build candidate updates WITH grad tracking.
                v_try = torch.clamp(v.unsqueeze(0) + alphas.view(T, 1, 1) * dv.unsqueeze(0), v_min, v_max)
                th_try = (th.unsqueeze(0) + alphas.view(T, 1, 1) * dth.unsqueeze(0) + math.pi) % (2 * math.pi) - math.pi

                # Score candidates WITHOUT grad (decision only).
                with torch.no_grad():
                    # One broadcasted complex matmul evaluates all T candidates
                    # for all B samples, replacing five Python-dispatched calls.
                    F_all = _batched_mismatch_inf_norm(
                        Y, v_try, th_try, P_set.unsqueeze(0), Q_set.unsqueeze(0),
                        slack_mask.unsqueeze(0), pv_mask.unsqueeze(0),
                    )
                    c1 = 1e-4
                    cond = F_all <= (1.0 - c1 * alphas[:, None]) * F0[None, :]
                    candidate_ids = torch.arange(T, device=device)[:, None].expand(T, B)
                    first = torch.where(cond, candidate_ids, T).amin(dim=0)
                    fallback = (first == T) & (F_all[-1] < F0)
                    accepted = (first < T) | fallback
                    first = torch.where(first < T, first, torch.full_like(first, T - 1))

                b_idx = torch.arange(B, device=device)
                chosen_v = v_try[first, b_idx]
                chosen_th = th_try[first, b_idx]
                chosen_a = alphas[first].view(B, 1)
                v = torch.where(accepted[:, None], chosen_v, v)
                th = torch.where(accepted[:, None], chosen_th, th)
                m = torch.where(accepted[:, None, None], m + chosen_a.unsqueeze(-1) * dm, m)
            else:
                th = (th + dth + math.pi) % (2 * math.pi) - math.pi
                v = torch.clamp(v + dv, 0.75, 1.2)
                m = m + dm

            if self.pinn:
                term = (self.gamma ** (self.K - 1 - k)) * ((DP ** 2 + DQ ** 2).mean())
                phys_terms.append(term)

        out = torch.stack([v, th], dim=-1)
        if self.pinn:
            phys_loss = torch.sum(torch.stack(phys_terms))
            return out, phys_loss
        else :
            return out
