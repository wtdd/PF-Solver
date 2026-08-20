from torch_geometric.nn import GATConv
import torch
import torch.nn as nn
import torch.nn.functional as F

class RJMNC(nn.Module):

    def __init__(
        self,
        in_channels=6,
        edge_dim=6,
        hidden_channels=64,
        out_channels=2,
        num_node_types=2,
        type_emb_dim=4,
        num_nodes=13,
        node_emb_dim=4,
        max_steps=4,
        step_emb_dim=4,
        local_hidden_dim=32,
        gat_init_scale=0.1,
        dropout=0.01,

        # GAT layer parameters
        heads1=2,
        heads2=2,
        heads3=2,
        gat2_channels=None,

        # MLP head parameters
        lin_hidden1=64,
        lin_hidden2=32
    ):
        super().__init__()

        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")

        self.num_nodes = num_nodes
        self.max_steps = max_steps
        # These integer index tensors are identical for every pass of a fixed
        # batch size. Keeping a non-persistent runtime cache avoids rebuilding
        # them at every correction step without changing any model values.
        self._node_index_cache = {}
        self._step_index_cache = {}
        self._index_cache_max_shapes = 8

        # --------------------------------------------------
        # 1. Embedding layers
        # --------------------------------------------------
        self.type_embedding = nn.Embedding(
            num_embeddings=num_node_types,
            embedding_dim=type_emb_dim
        )

        self.node_embedding = nn.Embedding(
            num_embeddings=num_nodes,
            embedding_dim=node_emb_dim
        )

        self.step_embedding = nn.Embedding(
            num_embeddings=max_steps,
            embedding_dim=step_emb_dim
        )
        # Raw node features + PQ/PV embedding + node embedding + phase conditioning
        input_dim = in_channels + type_emb_dim + node_emb_dim + step_emb_dim

        # --------------------------------------------------
        # 2. Local branch
        # --------------------------------------------------
        self.local_head = nn.Sequential(
            nn.Linear(input_dim, local_hidden_dim),
            nn.SiLU(),
            nn.Linear(local_hidden_dim, out_channels)
        )

        # --------------------------------------------------
        # 3. GAT branch scale
        # --------------------------------------------------
        self.gat_scale = nn.Parameter(
            torch.tensor(gat_init_scale, dtype=torch.float32)
        )
        # Keep legacy behavior at initialization: effective scale starts at 1.0,
        # while the parameter can still learn to amplify or damp the GAT branch.
        self.gat_scale_base = 1.0 - float(gat_init_scale)

        # --------------------------------------------------
        # 4. Automatically determine the dimensions of each GAT layer
        # --------------------------------------------------
        if gat2_channels is None:
            gat2_channels = hidden_channels * 2

        gat1_out_dim = hidden_channels * heads1      # concat=True
        gat2_out_dim = gat2_channels * heads2        # concat=True
        gat3_out_dim = hidden_channels               # concat=False

        # --------------------------------------------------
        # 5. GAT branch
        # --------------------------------------------------
        self.gat1 = GATConv(
            input_dim,
            hidden_channels,
            heads=heads1,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=False
        )

        self.gat2 = GATConv(
            gat1_out_dim,
            gat2_channels,
            heads=heads2,
            concat=True,
            edge_dim=edge_dim,
            add_self_loops=False
        )

        self.gat3 = GATConv(
            gat2_out_dim,
            hidden_channels,
            heads=heads3,
            concat=False,
            edge_dim=edge_dim,
            add_self_loops=False
        )

        # --------------------------------------------------
        # 6. Residual and normalization
        # --------------------------------------------------
        self.res3 = nn.Linear(input_dim, gat3_out_dim)

        self.norm1 = nn.LayerNorm(gat1_out_dim)
        self.norm2 = nn.LayerNorm(gat2_out_dim)
        self.norm3 = nn.LayerNorm(gat3_out_dim)

        self.dropout = nn.Dropout(dropout)

        # --------------------------------------------------
        # 7. GAT output MLP head: three linear layers
        # --------------------------------------------------
        self.lin1 = nn.Linear(gat3_out_dim, lin_hidden1)
        self.lin2 = nn.Linear(lin_hidden1, lin_hidden2)
        self.lin3 = nn.Linear(lin_hidden2, out_channels)

    def _apply(self, fn, recurse=True):
        # Device/dtype moves invalidate runtime-only device-indexed caches.
        self._node_index_cache.clear()
        self._step_index_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def forward(self, data, step=0):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Ablation: zero the four Jacobian edge features (without-Jacobian variant)
        # edge_attr = edge_attr.clone()
        # edge_attr[:, 2:6] = 0.0

        total_nodes = x.size(0)

        # --------------------------------------------------
        # 1. PQ/PV type embedding
        # data.masks: PQ=1, PV=2
        # type_idx:   PQ=0, PV=1
        # --------------------------------------------------
        type_idx = data.masks.squeeze().long() - 1
        type_emb = self.type_embedding(type_idx)

        # --------------------------------------------------
        # 2. Node index embedding
        # --------------------------------------------------
        if hasattr(data, "num_graphs"):
            num_graphs = int(data.num_graphs)
            non_slack = total_nodes // num_graphs
        else:
            non_slack = self.num_nodes

        if non_slack != self.num_nodes:
            raise ValueError(
                f"node_embedding num_nodes={self.num_nodes}, "
                f"but current graph has non_slack={non_slack}. "
                f"Please set num_nodes correctly."
            )

        device_key = (x.device.type, x.device.index, total_nodes)
        node_idx = self._node_index_cache.get(device_key)
        if node_idx is None:
            # Keep cached indices as ordinary tensors even if the first call
            # happens inside inference_mode; embedding backward may save them
            # when the same batch shape is later used for training.
            with torch.inference_mode(False):
                node_idx = torch.arange(
                    total_nodes,
                    device=x.device,
                    dtype=torch.long
                ) % self.num_nodes
            if len(self._node_index_cache) >= self._index_cache_max_shapes:
                oldest_key = next(iter(self._node_index_cache))
                self._node_index_cache.pop(oldest_key)
                for cached_step_key in list(self._step_index_cache):
                    if cached_step_key[:3] == oldest_key:
                        self._step_index_cache.pop(cached_step_key)
            self._node_index_cache[device_key] = node_idx

        node_emb = self.node_embedding(node_idx)

        # --------------------------------------------------
        # 3. Phase conditioning
        # --------------------------------------------------
        if isinstance(step, int):
            step = max(0, min(step, self.max_steps - 1))
            step_key = device_key + (step,)
            step_idx = self._step_index_cache.get(step_key)
            if step_idx is None:
                with torch.inference_mode(False):
                    step_idx = torch.full(
                        (total_nodes,),
                        fill_value=step,
                        dtype=torch.long,
                        device=x.device
                    )
                self._step_index_cache[step_key] = step_idx
        else:
            step_idx = step.to(device=x.device, dtype=torch.long)

            if step_idx.dim() == 0:
                step_idx = step_idx.view(1).expand(total_nodes)

            elif step_idx.numel() == 1:
                step_idx = step_idx.expand(total_nodes)

            elif hasattr(data, "batch") and step_idx.numel() == data.num_graphs:
                step_idx = step_idx[data.batch]

            elif step_idx.numel() == total_nodes:
                pass

            else:
                raise ValueError(
                    f"Invalid step shape: got {step_idx.shape}, "
                    f"expected scalar, [num_graphs], or [total_nodes]."
                )

            step_idx = torch.clamp(
                step_idx,
                min=0,
                max=self.max_steps - 1
            )
        step_emb = self.step_embedding(step_idx)
        # --------------------------------------------------
        # 4. Input concat
        # --------------------------------------------------

        # Ablation: remove phase conditioning
        # step_emb = x.new_zeros((total_nodes, self.step_embedding.embedding_dim))
        x_input = torch.cat([x, type_emb, node_emb, step_emb], dim=-1)

        # --------------------------------------------------
        # 5. Local branch
        # --------------------------------------------------
        local_out = self.local_head(x_input)

        # --------------------------------------------------
        # 6. GAT branch
        # --------------------------------------------------
        x1 = self.gat1(x_input, edge_index, edge_attr)
        x1 = self.norm1(x1)
        x1 = F.elu(x1)
        x1 = self.dropout(x1)

        x2 = self.gat2(x1, edge_index, edge_attr)
        x2 = self.norm2(x2)
        x2 = F.elu(x2)
        x2 = self.dropout(x2)

        x3 = self.gat3(x2, edge_index, edge_attr)

        # hidden-level residual
        x3 = self.norm3(x3 + self.res3(x_input))
        x3 = F.elu(x3)
        x3 = self.dropout(x3)

        # --------------------------------------------------
        # 7. Three-layer Linear head
        # --------------------------------------------------
        x3 = self.lin1(x3)
        x3 = F.elu(x3)
        x3 = self.dropout(x3)

        x3 = self.lin2(x3)
        x3 = F.elu(x3)
        x3 = self.dropout(x3)

        gat_out = self.lin3(x3)

        # --------------------------------------------------
        # 8. Output-level decomposition
        # --------------------------------------------------
        out = local_out + (self.gat_scale + self.gat_scale_base) * gat_out

        # Do not predict voltage-magnitude corrections for PV buses.
        pq_mask = (data.masks.squeeze() == 1).to(dtype=out.dtype)

        out = torch.stack([out[:, 0], out[:, 1] * pq_mask], dim=1)

        return out
