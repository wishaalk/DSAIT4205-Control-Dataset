"""
Generates the SBM control dataset for testing GSL robustness to structural noise.
Run: python generate_control_dataset.py
"""

import numpy as np
import scipy.sparse as sp
import pickle
from pathlib import Path


# === Configuration ===
N_PER_CLUSTER = 100
N_CLUSTERS = 5
P_IN = 0.3
P_OUT = 0.01
N_FEATURES = 16
SIGMA = 1.5
CENTROID_SCALE = 0.5
NOISE_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "data"


def generate_sbm_graph(n_per_cluster, n_clusters, p_in, p_out, seed):
    """Generate an SBM graph. Returns sparse adjacency + labels."""
    rng = np.random.default_rng(seed)
    n = n_per_cluster * n_clusters
    labels = np.repeat(np.arange(n_clusters), n_per_cluster)

    rows, cols = [], []
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                rows.extend([i, j])
                cols.extend([j, i])

    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return adj, labels


def generate_node_features(labels, n_features, sigma, centroid_scale, seed):
    """Generate Gaussian features: per-cluster centroid + noise."""
    rng = np.random.default_rng(seed)
    n_clusters = len(np.unique(labels))
    centroids = rng.normal(0, centroid_scale, size=(n_clusters, n_features))

    features = np.array([
        centroids[labels[i]] + rng.normal(0, sigma, size=n_features)
        for i in range(len(labels))
    ])
    return features


def corrupt_graph(adj, noise_rate, seed):
    """Remove edges with probability noise_rate, add back the same number of random edges."""
    rng = np.random.default_rng(seed)
    n = adj.shape[0]

    # Work with upper triangle (symmetric graph)
    edges = np.array(sp.triu(adj, k=1).nonzero()).T
    n_edges = len(edges)

    # Remove edges
    keep_mask = rng.random(n_edges) > noise_rate
    kept_edges = edges[keep_mask]
    n_to_add = n_edges - len(kept_edges)

    # Add random non-edges
    adj_set = set(map(tuple, edges))
    added_edges = []
    attempts = 0
    while len(added_edges) < n_to_add and attempts < n_to_add * 10:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        edge = (min(i, j), max(i, j))
        if edge not in adj_set:
            added_edges.append(edge)
            adj_set.add(edge)
        attempts += 1

    # Build new adjacency
    all_edges = np.vstack([kept_edges, added_edges]) if added_edges else kept_edges
    rows = np.concatenate([all_edges[:, 0], all_edges[:, 1]])
    cols = np.concatenate([all_edges[:, 1], all_edges[:, 0]])
    return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def compute_structure_recovery(learned_adj, true_adj):
    """Precision, recall, F1 of edges vs ground truth."""
    learned_set = set(zip(*sp.triu(learned_adj, k=1).nonzero()))
    true_set = set(zip(*sp.triu(true_adj, k=1).nonzero()))

    if not learned_set:
        return 0.0, 0.0, 0.0

    tp = len(learned_set & true_set)
    precision = tp / len(learned_set)
    recall = tp / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Generate graph
    print("Generating SBM graph...")
    adj_clean, labels = generate_sbm_graph(N_PER_CLUSTER, N_CLUSTERS, P_IN, P_OUT, SEED)
    print(f"  {adj_clean.shape[0]} nodes, {adj_clean.nnz // 2} edges")

    # 2. Generate features
    print("Generating node features...")
    features = generate_node_features(labels, N_FEATURES, SIGMA, CENTROID_SCALE, SEED)

    # 3. Save clean data
    np.save(OUTPUT_DIR / "features.npy", features)
    np.save(OUTPUT_DIR / "labels.npy", labels)
    sp.save_npz(OUTPUT_DIR / "adj_clean.npz", adj_clean)

    # 4. Generate and save corrupted graphs
    print("Corrupting graph at each noise rate...")
    for rate in NOISE_RATES:
        corrupted = adj_clean if rate == 0.0 else corrupt_graph(adj_clean, rate, SEED)
        sp.save_npz(OUTPUT_DIR / f"adj_noisy_{rate:.1f}.npz", corrupted)
        p, r, f1 = compute_structure_recovery(corrupted, adj_clean)
        print(f"  noise={rate:.0%}: {corrupted.nnz // 2} edges | P={p:.3f} R={r:.3f} F1={f1:.3f}")

    # 5. Save train/val/test split
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(labels))
    n_train = int(len(labels) * 0.1)
    n_val = int(len(labels) * 0.1)

    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)
    train_mask[indices[:n_train]] = True
    val_mask[indices[n_train:n_train + n_val]] = True
    test_mask[indices[n_train + n_val:]] = True

    with open(OUTPUT_DIR / "splits.pkl", "wb") as f:
        pickle.dump({"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask}, f)

    # 6. Save metadata
    metadata = {
        "n_nodes": len(labels),
        "n_clusters": len(np.unique(labels)),
        "n_features": features.shape[1],
        "n_edges_clean": adj_clean.nnz // 2,
        "noise_rates": NOISE_RATES,
    }
    with open(OUTPUT_DIR / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print(f"\nDone. Saved to {OUTPUT_DIR}/")
