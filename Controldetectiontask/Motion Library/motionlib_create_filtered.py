#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
motionlib_create_filtered.py  – updated for 5-second snippets at 60Hz
─────────────────────────────────────────────────────────────────────────
• Loads every *.csv in data_dir
• Slices traces into 5-s (300-frame) snippets
• Drops 'all-pause' snippets  (pause_ratio == 1)
• Drops snippets whose path length < 20 % of that donor's median
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
DESIRED_PER   = 1000   # target per cluster (will clamp to available)
SNIP_LEN      = 300    # 5 s at 60 Hz
LOWMOVE_FRAC  = 0.20   # snippets with path < 20 % of donor‐median are dropped
K_CLUST       = 4      # nominal number of clusters (reduced if M < 4)

# ─── locate CSVs ────────────────────────────────────────
data_dir  = pathlib.Path("/Users/simonknogler/Desktop/PhD/WP1/Controldetectiontask/Motion Library/data")
csv_files = sorted(data_dir.glob("*.csv"))
# Filter out analysis files
csv_files = [f for f in csv_files if not any(x in f.name for x in ["analysis", "trial_details"])]
if not csv_files:
    sys.exit(f"No CSV files found in {data_dir}")

print(f"Found {len(csv_files)} CSV files to process")

# ─── gather snippets, features, donor IDs ───────────────
all_snips, all_feats, donor_ids = [], [], []

for csvf in csv_files:
    print(f"Processing {csvf.name}...")
    
    try:
        df = pd.read_csv(csvf)
    except Exception as e:
        print(f"  Error reading {csvf.name}: {e}")
        continue
    
    # Check if this is a participant data file
    if 'trial' not in df.columns:
        print(f"  Skipping {csvf.name} - not a participant data file")
        continue
    
    if 'x' not in df.columns or 'y' not in df.columns:
        print(f"  Skipping {csvf.name} - missing x,y columns")
        continue
    
    # Get participant ID from the Participant column or filename
    if 'Participant' in df.columns and df['Participant'].notna().any():
        participant_id = str(df['Participant'].dropna().iloc[0]).strip()
    else:
        # Extract from filename if Participant column is empty
        participant_id = csvf.stem.replace('PARTICIPANT_buildermolib_', '')
    
    if not participant_id or participant_id == 'nan':
        participant_id = csvf.stem
    
    # Remove NaN trials and sort by trial number
    df_clean = df.dropna(subset=['trial'])
    
    # Group by trial and process each trial as a snippet
    trial_groups = df_clean.groupby('trial')
    
    for trial_num, trial_data in trial_groups:
        trial_data = trial_data.reset_index(drop=True)
        
        print(f"  Trial {trial_num}: {len(trial_data)} samples")
        
        # Check if we have enough samples for a full snippet
        if len(trial_data) < SNIP_LEN:
            print(f"    Insufficient samples ({len(trial_data)} < {SNIP_LEN})")
            continue
        
        # Take the first SNIP_LEN samples from this trial
        coords = trial_data[["x", "y"]].values[:SNIP_LEN].astype(np.float32)
        
        # Reshape into (1, 300, 2) - one snippet per trial
        snips = coords.reshape(1, SNIP_LEN, 2)
        n_snips = 1

        # Compute per-frame deltas for speed & angle features
        diff   = np.diff(snips, axis=1)                  # shape (1, 299, 2)
        dx, dy = diff[:, :, 0], diff[:, :, 1]             # each (1, 299)
        speed  = np.sqrt(dx**2 + dy**2)                   # magnitude per frame
        theta  = np.arctan2(dy, dx)                       # angle per frame

        # Extract six features per snippet:
        # 1. mean(frame-by-frame speed)
        # 2. SD(frame-by-frame speed)
        # 3. mean(unwrapped angle)
        # 4. SD(unwrapped angle)
        # 5. path straightness = total speed sum / net displacement
        # 6. pause_fraction = (# frames where speed < 0.1) / 299
        feats = np.column_stack([
            speed.mean(axis=1),
            speed.std(axis=1),
            np.unwrap(theta, axis=1).mean(axis=1),
            np.unwrap(theta, axis=1).std(axis=1),
            speed.sum(axis=1) / (np.linalg.norm(snips[:, -1] - snips[:, 0], axis=1) + 1e-9),
            (speed < 0.1).sum(axis=1) / speed.shape[1]
        ]).astype(np.float32)  # shape (1, 6)

        all_snips.append(snips)    # raw absolute positions (1, 300, 2)
        all_feats.append(feats)    # feature matrix (1, 6)
        donor_ids.extend([participant_id] * n_snips)
        
        print(f"    Added snippet from trial {trial_num}")

if not all_snips:
    sys.exit("No valid snippets found in any CSV.")

# Concatenate across all CSVs
snips     = np.concatenate(all_snips, axis=0)     # shape (M, 300, 2)
feats     = np.concatenate(all_feats, axis=0)     # shape (M, 6)
donor_ids = np.array(donor_ids)                   # length M
M         = snips.shape[0]
print(f"\nLoaded {M} snippets from {len(csv_files)} CSV file(s)")

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
    if np.sum(low):
        print(f"  {donor}: dropping {np.sum(low)} snippets below {thr:.1f} (med={med:.1f})")
        keep2[low] = False

snips, feats, donor_ids = snips[keep2], feats[keep2], donor_ids[keep2]
print(f"{snips.shape[0]} remain after low-movement filter")

# ─── clustering ───────────────────────────────────────────
print(f"\nClustering {snips.shape[0]} snippets into K={K_CLUST} groups …")
scaler = StandardScaler()
feats_scaled = scaler.fit_transform(feats)

K = min(K_CLUST, snips.shape[0])  # can't have more clusters than data points
if K < 2:
    print("Warning: Too few snippets for clustering. Using single cluster.")
    labels = np.zeros(snips.shape[0], dtype=int)
else:
    km = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = km.fit_predict(feats_scaled)
    try:
        sil_score = silhouette_score(feats_scaled, labels)
        print(f"Silhouette score: {sil_score:.3f}")
    except:
        print("Could not compute silhouette score")

cluster_sizes = np.bincount(labels)
print(f"Cluster sizes: {cluster_sizes}")

# ─── proportional sampling ──────────────────────────────
print(f"\nUsing proportional sampling to preserve all {snips.shape[0]} trajectories ...")

# Use all available trajectories (no sampling limit)
sampled_indices = list(range(snips.shape[0]))

core_pool_snips  = snips[sampled_indices]
core_pool_feats  = feats[sampled_indices] 
core_pool_labels = labels[sampled_indices]

print(f"Final pool: {core_pool_snips.shape[0]} trajectories (100% of quality-filtered data)")
print(f"Cluster distribution maintained:")
final_cluster_sizes = np.bincount(core_pool_labels)
for c in range(K):
    original_size = cluster_sizes[c]
    final_size = final_cluster_sizes[c] if c < len(final_cluster_sizes) else 0
    percentage = (final_size / original_size * 100) if original_size > 0 else 0
    print(f"  Cluster {c}: {final_size}/{original_size} trajectories ({percentage:.1f}%)")

# ─── convert absolute positions → velocities ─────────────
print("\nConverting to velocity representation …")
core_pool_vels = np.diff(core_pool_snips, axis=1)  # shape (N, 299, 2)

# ─── save outputs ─────────────────────────────────────────
np.save("core_pool.npy", core_pool_vels)
np.save("core_pool_feats.npy", core_pool_feats)  
np.save("core_pool_labels.npy", core_pool_labels)

# Save scaler parameters for downstream use
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist()
}
with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)

# Save cluster centroids if we have a fitted KMeans
if K >= 2:
    cluster_centroids = km.cluster_centers_.tolist()
    with open('cluster_centroids.json', 'w') as f:
        json.dump(cluster_centroids, f, indent=2)

print(f"\nSaved:")
print(f"  core_pool.npy            : shape {core_pool_vels.shape}")
print(f"  core_pool_feats.npy      : shape {core_pool_feats.shape}")
print(f"  core_pool_labels.npy     : shape {core_pool_labels.shape}")
print(f"  scaler_params.json       : normalization parameters")
if K >= 2:
    print(f"  cluster_centroids.json   : cluster centers")
print("Motion library creation complete!")
