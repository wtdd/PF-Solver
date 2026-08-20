import os
import csv
from collections import defaultdict
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


def _common_dir(data_path):
    return os.path.join(data_path, "")


def _has_common_dataset(data_path):
    common_dir = _common_dir(data_path)
    required = ("meta.csv", "bus_static.csv", "ybus.csv", "bus_state.csv", "jacobian_start.csv")
    return all(os.path.exists(os.path.join(common_dir, name)) for name in required)


def _as_bool(value):
    text = str(value).strip().lower()
    return text in ("1", "true", "yes")


def _read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_bus_static_common(common_dir):
    rows = _read_csv_rows(os.path.join(common_dir, "bus_static.csv"))
    if len(rows) == 0:
        raise ValueError(f"Empty bus_static.csv: {common_dir}")
    return rows


# Get G/B admittance maps in tensor form.
def loadAdmittance(data_path, device):
    if _has_common_dataset(data_path):
        common_dir = _common_dir(data_path)
        bus_rows = _read_bus_static_common(common_dir)
        bus_total = max(int(row["bus0"]) for row in bus_rows) + 1

        slack_rows = [row for row in bus_rows if _as_bool(row["is_slack"])]
        if len(slack_rows) != 1:
            raise ValueError(f"Expected exactly one slack bus in {common_dir}/bus_static.csv, got {len(slack_rows)}")
        slack_row = slack_rows[0]

        Gij = torch.zeros((bus_total, bus_total), dtype=torch.float32)
        Bij = torch.zeros((bus_total, bus_total), dtype=torch.float32)
        for row in _read_csv_rows(os.path.join(common_dir, "ybus.csv")):
            i = int(row["from_bus0"])
            j = int(row["to_bus0"])
            Gij[i, j] = float(row["g"])
            Bij[i, j] = float(row["b"])

        slack_node = torch.tensor(
            [
                float(slack_row["bus_id"]),
                float(slack_row["va_init_rad"]),
                float(slack_row["vm_init"]),
            ],
            dtype=torch.float32,
        )
        return Gij.to(device), Bij.to(device), slack_node.to(device)

    file_path = f"{data_path}/Admittance.txt"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]

    slack_node = lines[-1].split(",")
    slack_node = [float(x) for x in slack_node]

    Gij, Bij = {}, {}
    for i, line in enumerate(lines[:-1], start=1):
        Gij.setdefault(i, {})
        Bij.setdefault(i, {})
        fields = [s for s in line.split("#") if s]
        for item in fields:
            parts = item.split(",")
            if len(parts) != 3:
                raise ValueError(f"Invalid Admittance format at row {i}: {item}")
            to = int(parts[0])
            gij = float(parts[1])
            bij = float(parts[2])
            Gij[i][to] = gij
            Bij[i][to] = bij

    def nested_dict_to_tensor(dct, dev):
        row_keys = sorted(dct.keys())
        col_keys = sorted({ck for rv in dct.values() for ck in rv.keys()})
        n_row, n_col = len(row_keys), len(col_keys)
        row_idx = {k: i for i, k in enumerate(row_keys)}
        col_idx = {k: i for i, k in enumerate(col_keys)}
        mat = torch.zeros(n_row, n_col, dtype=torch.float32)
        for r, inner in dct.items():
            for c, val in inner.items():
                mat[row_idx[r], col_idx[c]] = val
        return mat.to(dev)

    Gij = nested_dict_to_tensor(Gij, device)
    Bij = nested_dict_to_tensor(Bij, device)
    slack_node = torch.tensor(slack_node, dtype=torch.float32).to(device)
    return Gij, Bij, slack_node


def _load_common_data(data_path):
    common_dir = _common_dir(data_path)
    meta_rows = sorted(_read_csv_rows(os.path.join(common_dir, "meta.csv")), key=lambda row: int(row["sample_id"]))

    bus_by_sample = defaultdict(list)
    for row in _read_csv_rows(os.path.join(common_dir, "bus_state.csv")):
        bus_by_sample[int(row["sample_id"])].append(row)

    jac_by_sample = defaultdict(list)
    for row in _read_csv_rows(os.path.join(common_dir, "jacobian_start.csv")):
        jac_by_sample[int(row["sample_id"])].append(row)

    node_data, line_data, output_data = [], [], []
    node_data_test, line_data_test, output_data_test = [], [], []
    level_data_test, valid_label_test, ill_label_test = [], [], []

    for meta in meta_rows:
        sample_id = int(meta["sample_id"])
        split = meta["split"].strip().lower()

        bus_rows = [row for row in bus_by_sample[sample_id] if int(row["nonslack0"]) >= 0]
        bus_rows.sort(key=lambda row: int(row["nonslack0"]))
        if len(bus_rows) == 0:
            raise ValueError(f"Common dataset sample {sample_id} has no non-slack bus rows.")

        node_sample = []
        output_sample = []
        for row in bus_rows:
            node_id = float(row["nonslack0"])
            bus_type = int(row["type"])
            dq_start = float(row["dq_start"]) if bus_type == 1 else 0.0
            dv_label = float(row["dv_label"]) if bus_type == 1 else 0.0

            node_sample.append(
                [
                    node_id,
                    float(row["p_calc_start"]),
                    float(row["q_calc_start"]),
                    float(row["dp_start"]),
                    dq_start,
                    float(row["va_start_rad"]),
                    float(row["vm_start"]),
                    float(bus_type),
                ]
            )
            output_sample.append([node_id, float(row["dtheta_label"]), dv_label])

        line_rows = [
            row
            for row in jac_by_sample[sample_id]
            if int(row["from_nonslack0"]) >= 0 and int(row["to_nonslack0"]) >= 0
        ]
        line_rows.sort(key=lambda row: (int(row["from_nonslack0"]), int(row["to_nonslack0"])))
        if len(line_rows) == 0:
            raise ValueError(f"Common dataset sample {sample_id} has no non-slack Jacobian rows.")

        line_sample = [
            [
                float(row["from_nonslack0"]),
                float(row["to_nonslack0"]),
                float(row["g"]),
                float(row["b"]),
                float(row["H"]),
                float(row["M"]),
                float(row["K"]),
                float(row["L"]),
            ]
            for row in line_rows
        ]

        if split == "train":
            node_data.append(node_sample)
            line_data.append(line_sample)
            output_data.append(output_sample)
        elif split == "test":
            source = meta.get("source", "")
            # Legacy schema label: these N-R PF hard-convergence cases combine
            # difficult initial states with stressed P/Q operating points near the
            # feasibility/stability boundary, where the Jacobian may be poorly conditioned.
            # Selection is based on N-R behavior, not an explicit condition-number threshold.
            is_ill_conditioned = str(source).strip().lower() == "ill-conditioned"
            node_data_test.append(node_sample)
            line_data_test.append(line_sample)
            output_data_test.append(output_sample)
            level_data_test.append(
                [
                    float(meta["x_low"]),
                    float(meta["x_high"]),
                    float(meta["pq_low"]),
                    float(meta["pq_high"]),
                    float(meta["x_signed"]),
                    float(meta["pq_signed"]),
                ]
            )
            valid_label_test.append(1 if _as_bool(meta["valid_label"]) else 0)
            ill_label_test.append(1 if is_ill_conditioned else 0)
        else:
            raise ValueError(f"Unknown split '{meta['split']}' in sample {sample_id}.")

    if not (len(node_data) == len(line_data) == len(output_data)):
        raise ValueError("Common train sample count mismatch.")
    if not (len(node_data_test) == len(line_data_test) == len(output_data_test) == len(level_data_test) == len(valid_label_test) == len(ill_label_test)):
        raise ValueError("Common test sample count mismatch.")

    num_samples = len(line_data)
    rng = np.random.default_rng(42)
    idx = np.arange(num_samples)
    rng.shuffle(idx)

    node_misorder = [node_data[i] for i in idx]
    line_misorder = [line_data[i] for i in idx]
    output_misorder = [output_data[i] for i in idx]

    return (node_misorder, line_misorder, output_misorder, node_data_test, line_data_test, output_data_test, level_data_test, valid_label_test, ill_label_test,)


def Standardization(node_data, line_data, output_data, node_test, line_test, output_test):
    nodes = np.asarray(node_data, dtype=np.float32).copy()
    lines = np.asarray(line_data, dtype=np.float32).copy()
    outs = np.asarray(output_data, dtype=np.float32).copy()

    nodes_test = np.asarray(node_test, dtype=np.float32).copy()
    lines_test = np.asarray(line_test, dtype=np.float32).copy()
    outs_test = np.asarray(output_test, dtype=np.float32).copy()

    num_samples = len(nodes)
    train_size = int(num_samples * 0.9)

    train_nodes_all = nodes[:train_size, :, 1:7].reshape(-1, 6)
    train_lines_all = lines[:train_size, :, 2:8].reshape(-1, 6)
    train_outs_all = outs[:train_size, :, 1:3].reshape(-1, 2)

    nodes_mean, nodes_std = train_nodes_all.mean(axis=0), train_nodes_all.std(axis=0)
    lines_mean, lines_std = train_lines_all.mean(axis=0), train_lines_all.std(axis=0)
    outs_mean, outs_std = train_outs_all.mean(axis=0), train_outs_all.std(axis=0)

    outs_mean = np.zeros_like(outs_mean)
    nodes_std[nodes_std == 0] = 1.0
    lines_std[lines_std == 0] = 1.0
    outs_std[outs_std == 0] = 1.0

    nodes[:, :, 1:7] = (nodes[:, :, 1:7] - nodes_mean) / nodes_std
    lines[:, :, 2:8] = (lines[:, :, 2:8] - lines_mean) / lines_std
    outs[:, :, 1:3] = (outs[:, :, 1:3] - outs_mean) / outs_std

    nodes_test[:, :, 1:7] = (nodes_test[:, :, 1:7] - nodes_mean) / nodes_std
    lines_test[:, :, 2:8] = (lines_test[:, :, 2:8] - lines_mean) / lines_std
    outs_test[:, :, 1:3] = (outs_test[:, :, 1:3] - outs_mean) / outs_std

    return (nodes, lines, outs, nodes_test, lines_test, outs_test, nodes_mean, nodes_std, outs_mean, outs_std, lines_mean, lines_std,)


def construct_DATA(nodes, lines, outs, levels=None, valid_labels=None, ill_labels=None):
    data_list = []
    for i in range(len(nodes)):
        nodes_i, lines_i, outs_i = nodes[i], lines[i], outs[i]

        edge_index = torch.tensor(
            [[int(e[0]), int(e[1])] for e in lines_i], dtype=torch.long
        ).t().contiguous()
        edge_attr = torch.from_numpy(np.asarray([e[2:8] for e in lines_i], dtype=np.float32))

        x = torch.from_numpy(np.asarray([n[1:7] for n in nodes_i], dtype=np.float32))
        masks = torch.tensor([int(n[7]) for n in nodes_i], dtype=torch.long)

        y = torch.from_numpy(np.asarray([o[1:3] for o in outs_i], dtype=np.float32))

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.masks = masks

        if levels is not None:
            meta = torch.tensor(levels[i], dtype=torch.float32)
            data.level = meta
            data.x_low = meta[0].view(1)
            data.x_high = meta[1].view(1)
            data.pq_low = meta[2].view(1)
            data.pq_high = meta[3].view(1)
            data.x_signed = meta[4].view(1)
            data.pq_signed = meta[5].view(1)

        if valid_labels is not None:
            data.valid_label = torch.tensor([valid_labels[i]], dtype=torch.long)
        if ill_labels is not None:
            data.ill_conditioned = torch.tensor([ill_labels[i]], dtype=torch.long)

        data_list.append(data)

    return data_list


def _torch_load_cpu(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _processed_cache_exists(train_path, val_path, test_path, stats_path, meta_path):
    return (
        os.path.exists(train_path)
        and os.path.exists(val_path)
        and os.path.exists(test_path)
        and os.path.exists(stats_path)
        and os.path.exists(meta_path)
    )


def _is_heatmap_test_cache_valid(test_list):
    if len(test_list) == 0:
        return True
    sample = test_list[0]
    if not hasattr(sample, "level"):
        return False
    if int(sample.level.numel()) != 6:
        return False
    if not hasattr(sample, "valid_label"):
        return False
    if not hasattr(sample, "ill_conditioned"):
        return False
    return True


def _split_test_lists(test_list):
    heatmap_test_list = []
    ill_test_list = []
    for data in test_list:
        is_ill = bool(int(data.ill_conditioned.view(-1)[0].item())) if hasattr(data, "ill_conditioned") else False
        if is_ill:
            ill_test_list.append(data)
        else:
            heatmap_test_list.append(data)
    return heatmap_test_list, ill_test_list


def load_dataset_fast(datas_path, BatchSize, cache_dir=None, force_rebuild=False, test_batch_size=1, device="cpu"):
    print("Loading Train Data...")
    pin_memory = torch.device(device).type == "cuda"
    if cache_dir is None:
        cache_dir = os.path.join(datas_path, "processed_cache")

    train_cache_path = os.path.join(cache_dir, "train_list.pt")
    val_cache_path = os.path.join(cache_dir, "val_list.pt")
    test_cache_path = os.path.join(cache_dir, "test_list.pt")
    stats_cache_path = os.path.join(cache_dir, "stats.pt")
    meta_cache_path = os.path.join(cache_dir, "meta.pt")

    use_cache = False
    if (not force_rebuild) and _processed_cache_exists(train_cache_path, val_cache_path, test_cache_path, stats_cache_path, meta_cache_path,):
        print(f"Loading processed dataset cache from: {cache_dir}")
        train_list = _torch_load_cpu(train_cache_path)
        val_list = _torch_load_cpu(val_cache_path)
        test_list = _torch_load_cpu(test_cache_path)
        stats = _torch_load_cpu(stats_cache_path)
        meta = _torch_load_cpu(meta_cache_path)
        cache_version = int(meta.get("version", 0)) if isinstance(meta, dict) else 0

        if cache_version >= 4 and _is_heatmap_test_cache_valid(test_list):
            use_cache = True
        else:
            print("Processed cache is outdated for Common CSV data. Rebuilding cache...")

    if use_cache:
        nodes_mean = torch.as_tensor(stats["node_mean"], dtype=torch.float32, device=device)
        nodes_std = torch.as_tensor(stats["node_std"], dtype=torch.float32, device=device)
        outs_mean = torch.as_tensor(stats["output_mean"], dtype=torch.float32, device=device)
        outs_std = torch.as_tensor(stats["output_std"], dtype=torch.float32, device=device)
        lines_mean = torch.as_tensor(stats["lines_mean"], dtype=torch.float32, device=device)
        lines_std = torch.as_tensor(stats["lines_std"], dtype=torch.float32, device=device)

        train_loaders = DataLoader(train_list, batch_size=BatchSize, shuffle=True, pin_memory=pin_memory)
        val_loaders = DataLoader(val_list, batch_size=BatchSize, shuffle=False, pin_memory=pin_memory)
        heatmap_test_list, ill_test_list = _split_test_lists(test_list)
        test_loaders = DataLoader(heatmap_test_list, batch_size=test_batch_size, shuffle=False, pin_memory=pin_memory)
        ill_test_loaders = DataLoader(ill_test_list, batch_size=test_batch_size, shuffle=False, pin_memory=pin_memory)

        print(
            f"Loaded processed cache successfully: train={len(train_list)}, val={len(val_list)}, "
            f"heatmap_test={len(heatmap_test_list)}, nr_hard_test={len(ill_test_list)}, test_batch_size={test_batch_size}"
        )
        return (train_loaders, val_loaders, test_loaders, ill_test_loaders, nodes_mean, nodes_std, outs_mean, outs_std, lines_mean, lines_std,)

    (node_data, line_data, output_data, node_test, line_test, output_test, level_test, valid_label_test, ill_label_test,) = _load_common_data(datas_path)

    num_samples = len(line_data)
    print(f"Train dataset size: {num_samples}")
    train_size = int(num_samples * 0.9)

    (node_norm, line_norm, output_norm, nodes_test_norm, lines_test_norm, outs_test_norm, nodes_mean, nodes_std, outs_mean, outs_std, lines_mean, lines_std,) = Standardization(node_data, line_data, output_data, node_test, line_test, output_test)

    data_list = construct_DATA(node_norm, line_norm, output_norm)
    test_list = construct_DATA(nodes_test_norm, lines_test_norm, outs_test_norm, levels=level_test, valid_labels=valid_label_test, ill_labels=ill_label_test,)
    heatmap_test_list, ill_test_list = _split_test_lists(test_list)
    train_list = data_list[:train_size]
    val_list = data_list[train_size:]

    os.makedirs(cache_dir, exist_ok=True)
    stats = {"node_mean": nodes_mean, "node_std": nodes_std, "output_mean": outs_mean, "output_std": outs_std, "lines_mean": lines_mean, "lines_std": lines_std,}
    meta = {"version": 4, "datas_path": datas_path, "schema": "common_csv_v1", "train_size": train_size, "num_train": len(train_list), "num_val": len(val_list), "num_heatmap_test": len(heatmap_test_list), "num_ill_test": len(ill_test_list), "num_test": len(test_list),}

    print(f"Saving processed dataset cache to: {cache_dir}")
    torch.save(train_list, train_cache_path)
    torch.save(val_list, val_cache_path)
    torch.save(test_list, test_cache_path)
    torch.save(stats, stats_cache_path)
    torch.save(meta, meta_cache_path)
    print(f"Saved processed cache successfully: train={len(train_list)}, val={len(val_list)}, heatmap_test={len(heatmap_test_list)}, nr_hard_test={len(ill_test_list)}")

    train_loaders = DataLoader(train_list, batch_size=BatchSize, shuffle=True, pin_memory=pin_memory)
    val_loaders = DataLoader(val_list, batch_size=BatchSize, shuffle=False, pin_memory=pin_memory)
    test_loaders = DataLoader(heatmap_test_list, batch_size=test_batch_size, shuffle=False, pin_memory=pin_memory)
    ill_test_loaders = DataLoader(ill_test_list, batch_size=test_batch_size, shuffle=False, pin_memory=pin_memory)

    nodes_mean = torch.as_tensor(nodes_mean, dtype=torch.float32, device=device)
    nodes_std = torch.as_tensor(nodes_std, dtype=torch.float32, device=device)
    outs_mean = torch.as_tensor(outs_mean, dtype=torch.float32, device=device)
    outs_std = torch.as_tensor(outs_std, dtype=torch.float32, device=device)
    lines_mean = torch.as_tensor(lines_mean, dtype=torch.float32, device=device)
    lines_std = torch.as_tensor(lines_std, dtype=torch.float32, device=device)

    return (train_loaders, val_loaders, test_loaders, ill_test_loaders, nodes_mean, nodes_std, outs_mean, outs_std, lines_mean, lines_std,)

if __name__ == "__main__":
    BATCH_SIZE = 1024
    Data_path = "../Data/IEEE_14"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    (train_loaders, val_loaders, test_loaders, ill_test_loaders, nodes_mean, nodes_std, outs_mean, outs_std, lines_mean, lines_std,) = load_dataset_fast(Data_path, BATCH_SIZE, cache_dir=os.path.join(Data_path, "processed_cache"), force_rebuild=False, test_batch_size=1, device=device,)
