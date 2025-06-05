#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
motionlib_create_filtered.py  – updated to match build_core_pool_3s v21
─────────────────────────────────────────────────────────────────────────
• Loads every *.csv in data_dir
• Keeps only main‐block frames  (isPractice == 0)
• Slices traces into 3-s (180-frame) snippets
• Drops ‘all-pause’ snippets  (pause_ratio == 1)
• Drops snippets whose path length < 20 % of that donor’s median
• Clusters into K=6 in feature space (or fewer if not enough snippets)
• Samples proportionally to raw cluster sizes, but never more than exists
• Converts absolute positions → per-frame velocities before saving
• Writes core_pool.npy, core_pool_feats.npy, core_pool_labels.npy
• Saves scaler parameters and cluster centroids for downstream style-matching
"""

import numpy as np
import pandas as pd
import pathlib, sys, json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ─── parameters ─────────────────────────────────────────
DESIRED_PER   = 60     # target per cluster (will clamp to available)
SNIP_LEN      = 180    # 3 s at 60 Hz
LOWMOVE_FRAC  = 0.20   # snippets with path < 20 % of donor‐median are dropped
K_CLUST       = 6      # nominal number of clusters (reduced if M < 6)

# ─── locate CSVs ────────────────────────────────────────
data_dir  = pathlib.Path("/Users/simonknogler/Desktop/PhD/Buildermolib/data")
csv_files = sorted(data_dir.glob("*.csv"))
if not csv_files:
    sys.exit(f"No CSV files found in {data_dir}")

# ─── gather snippets, features, donor IDs ───────────────
all_snips, all_feats, donor_ids = [], [], []

for csvf in csv_files:
    df = pd.read_csv(csvf)
    df_main = df[df["isPractice"] == 0].reset_index(drop=True)
    coords  = df_main[["x", "y"]].to_numpy(dtype=np.float32)

    # Only keep full 3-s (180-frame) chunks
    usable = (len(coords) // SNIP_LEN) * SNIP_LEN
    if usable < SNIP_LEN:
        continue
    coords = coords[:usable]

    # Reshape into (n_snips, 180, 2)
    snips   = coords.reshape(-1, SNIP_LEN, 2)
    n_snips = snips.shape[0]

    # Compute per-frame deltas for speed & angle features
    diff   = np.diff(snips, axis=1)                  # shape (n_snips, 179, 2)
    dx, dy = diff[:, :, 0], diff[:, :, 1]             # each (n_snips, 179)
    speed  = np.sqrt(dx**2 + dy**2)                   # magnitude per frame
    theta  = np.arctan2(dy, dx)                       # angle per frame

    # Extract six features per snippet:
    # 1. mean(frame-by-frame speed)
    # 2. SD(frame-by-frame speed)
    # 3. mean(unwrapped angle)
    # 4. SD(unwrapped angle)
    # 5. path straightness = total speed sum / net displacement
    # 6. pause_fraction = (# frames where speed < 0.1) / 179
    feats = np.column_stack([
        speed.mean(axis=1),
        speed.std(axis=1),
        np.unwrap(theta, axis=1).mean(axis=1),
        np.unwrap(theta, axis=1).std(axis=1),
        speed.sum(axis=1) / (np.linalg.norm(snips[:, -1] - snips[:, 0], axis=1) + 1e-9),
        (speed < 0.1).sum(axis=1) / speed.shape[1]
    ]).astype(np.float32)  # shape (n_snips, 6)

    all_snips.append(snips)    # raw absolute positions (n_snips, 180, 2)
    all_feats.append(feats)    # feature matrix (n_snips, 6)

    donor = str(df.loc[0, "Participant"]).strip()
    donor_ids.extend([donor] * n_snips)

if not all_snips:
    sys.exit("No valid snippets found in any CSV.")

# Concatenate across all CSVs
snips     = np.concatenate(all_snips, axis=0)     # shape (M, 180, 2)
feats     = np.concatenate(all_feats, axis=0)     # shape (M, 6)
donor_ids = np.array(donor_ids)                   # length M
M         = snips.shape[0]
print(f"Loaded {M} snippets from {len(csv_files)} CSV file(s)")

# ─── drop all-pause snippets ─────────────────────────────
pause_ratio = feats[:, 5]
keep_mask   = pause_ratio < 1.0
if (drop := np.sum(~keep_mask)):
    print(f"Dropping {drop} all-pause snippets")
snips, feats, donor_ids = snips[keep_mask], feats[keep_mask], donor_ids[keep_mask]
print(f"{snips.shape[0]} remain after all-pause filter")

# ─── donor-specific low-movement exclusion ──────────────
print("\nApplying donor-specific low-movement filter …")
paths = np.sum(np.linalg.norm(np.diff(snips, axis=1), axis=2), axis=1)  # total path length per snippet
keep2 = np.ones_like(paths, dtype=bool)
for donor in np.unique(donor_ids):
    idx = donor_ids == donor
    med = np.median(paths[idx])
    thr = LOWMOVE_FRAC * med
    low = (paths < thr) & idx
    if low.any():
        print(f"  {donor}: dropping {low.sum()} / {idx.sum()} < {thr:.1f}px")
    keep2[low] = False

snips, feats, donor_ids = snips[keep2], feats[keep2], donor_ids[keep2]
print(f"{snips.shape[0]} remain after low-movement filter\n")

# ─── clustering & proportional sampling ──────────────────
K = min(K_CLUST, snips.shape[0])
print(f"Clustering into {K} groups")

# Z-score all six features before clustering
scaler = StandardScaler().fit(feats)
X      = scaler.transform(feats)  # shape (M, 6)

# Save scaler parameters for downstream style‐matching
scaler_params = {"mean": scaler.mean_.tolist(), "std": scaler.scale_.tolist()}
with open("scaler_params.json", "w") as f:
    json.dump(scaler_params, f)
print("Saved scaler parameters to scaler_params.json")

# Run K-means on the scaled features
kmeans = KMeans(n_clusters=K, n_init=20, random_state=0)
labels = kmeans.fit_predict(X)

# Save cluster centroids for downstream style‐matching
centroids = kmeans.cluster_centers_.tolist()  # list of K lists, each length 6
with open("cluster_centroids.json", "w") as f:
    json.dump(centroids, f)
print("Saved cluster centroids to cluster_centroids.json")

# Count raw cluster sizes
cluster_sizes = np.array([np.sum(labels == c) for c in range(K)])
total_snips   = cluster_sizes.sum()
print("Raw cluster sizes:", cluster_sizes)

# Ensure we never request more than exists in total
TOTAL_DESIRED = min(DESIRED_PER * K, total_snips)

# Compute desired sample counts proportional to raw sizes
props     = cluster_sizes / total_snips
n_samples = np.round(props * TOTAL_DESIRED).astype(int)

# Adjust rounding error so that sum(n_samples) == TOTAL_DESIRED
diff = TOTAL_DESIRED - n_samples.sum()
if diff > 0:
    for idx in np.argsort(-cluster_sizes)[:diff]:
        n_samples[idx] += 1
elif diff < 0:
    for idx in np.argsort(-cluster_sizes)[: -diff]:
        n_samples[idx] -= 1

print("Sampling counts per cluster:", n_samples)

# Draw snippet indices without replacement (clamp to available)
rng = np.random.default_rng(0)
core_indices = []
for c in range(K):
    idxs = np.where(labels == c)[0]
    needed = n_samples[c]
    if needed <= len(idxs):
        chosen = rng.choice(idxs, size=needed, replace=False)
    else:
        # Should not happen because TOTAL_DESIRED ≤ total_snips
        chosen = rng.choice(idxs, size=needed, replace=True)
    core_indices.extend(chosen)

# Build core_snips array (shape: (sum(n_samples), 180, 2)) of absolute positions
core_snips = snips[core_indices]  # still absolute positions

# ─── convert to per-frame velocities before saving ─────────────────────
# For each snippet, compute diff along axis=1 and prepend first row
core_vels = np.diff(core_snips, axis=1, prepend=core_snips[:, :1, :])
# core_vels has shape (N_core, 180, 2)

# Save core_pool.npy as float16 for compactness
out_path = data_dir.parent / "core_pool.npy"
np.save(out_path, core_vels.astype(np.float16))
print(f"Saved core_pool.npy {core_vels.shape} → {out_path}")

# Additionally save diagnostics: features and labels of the entire pool (pre-sampling)
np.save(data_dir.parent / "core_pool_feats.npy", feats.astype(np.float32))
np.save(data_dir.parent / "core_pool_labels.npy", labels.astype(np.int8))
print("Saved core_pool_feats.npy and core_pool_labels.npy")

# Compute and print silhouette score on full feature set
sil_score = silhouette_score(X, labels)
print("Mean silhouette score:", sil_score)
