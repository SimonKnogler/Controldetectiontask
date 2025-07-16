#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
control_detection_task_v2.py - 4-cluster version with participant-specific sampling
Windows-compatible version
────────────────────────────────────────────────────────────────────────────────────
Changes from v16.2:
• Uses 4 clusters instead of 6
• Assigns participants to their 2 closest clusters based on demo trial
• Samples equally from those 2 clusters for practice and test trials
• Loads cluster labels and features for cluster-specific sampling
"""

import os, sys, math, random, pathlib, datetime, atexit, hashlib, json, subprocess

# Check if we're running with the correct Python interpreter
def check_and_run_with_correct_python():
    # If psychopy import fails, try to find anaconda Python
    try:
        import numpy as np
        import pandas as pd
        from psychopy import visual, event, core, data, gui
        return False  # Continue with current interpreter
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Trying to find Python with required packages...")
        
        # Try common Python paths on different systems
        python_paths = [
            "/opt/anaconda3/bin/python",  # macOS
            "C:/Users/*/anaconda3/python.exe",  # Windows
            "C:/Users/*/miniconda3/python.exe",  # Windows
            "/usr/bin/python3",  # Linux
        ]
        
        for path in python_paths:
            if os.path.exists(path):
                print(f"Found Python at: {path}")
                result = subprocess.run([path] + sys.argv, check=False)
                sys.exit(result.returncode)
        
        print("Error: Python with required packages not found. Please install psychopy, numpy, and pandas.")
        sys.exit(1)

# Check interpreter and switch if needed
if check_and_run_with_correct_python():
    sys.exit(0)

# Import statements (will work after interpreter check)
import numpy as np
import pandas as pd
from psychopy import visual, event, core, data, gui

SIMULATE = True

# ───────────────────────────────────────────────────────
#  Global variable for kinematics data
# ───────────────────────────────────────────────────────
kinematics_data = []
kinematics_csv_path = ""

# ───────────────────────────────────────────────────────
#  Auto‐save on quit
# ───────────────────────────────────────────────────────
_saved = False
def _save():
    global _saved
    if not _saved:
        # Check if thisExp exists before trying to save
        if 'thisExp' in globals() and thisExp is not None:
            # Save main trial-by-trial data
            thisExp.saveAsWideText(csv_path)
            print("Main data auto‐saved ➜", csv_path)

            # Save frame-by-frame kinematics data
            if kinematics_data:
                kinematics_df = pd.DataFrame(kinematics_data)
                kinematics_df.to_csv(kinematics_csv_path, index=False)
                print("Kinematics data auto‐saved ➜", kinematics_csv_path)
        else:
            print("Experiment not initialized - no data to save")

        _saved = True
atexit.register(_save)

# ───────────────────────────────────────────────────────
#  Participant dialog
# ───────────────────────────────────────────────────────
expName = "ControlDetection_v2_4clusters"
expInfo = {"participant": "", "session": "001", "simulate": False}
dlg = gui.DlgFromDict(expInfo, order=["participant", "session", "simulate"], title=expName)
if not dlg.OK:
    core.quit()
SIMULATE = bool(expInfo.pop("simulate"))
if SIMULATE:
    expInfo["participant"] = "SIM"

# counter‐balance cue colours
low_col, high_col = random.choice([("blue","green"),("green","blue")])
expInfo["low_precision_colour"]  = low_col
expInfo["high_precision_colour"] = high_col

# ───────────────────────────────────────────────────────
#  Load motion library with cluster information
# ───────────────────────────────────────────────────────
# Use relative paths based on script location
script_dir = pathlib.Path(__file__).parent
LIB_NAME = script_dir / "Motion Library" / "core_pool.npy"
FEATS_NAME = script_dir / "Motion Library" / "core_pool_feats.npy"
LABELS_NAME = script_dir / "Motion Library" / "core_pool_labels.npy"

motion_pool = np.load(LIB_NAME)
snippet_features = np.load(FEATS_NAME)
snippet_labels = np.load(LABELS_NAME)

SNIP_LEN = motion_pool.shape[1]
TOTAL_SNIPS = motion_pool.shape[0]
K_CLUST = 4

print(f"Loaded {TOTAL_SNIPS} snippets × {SNIP_LEN} frames from {LIB_NAME}")
print(f"Cluster distribution: {np.bincount(snippet_labels)}")

# Load scaler parameters and cluster centroids
with open(script_dir / "Motion Library" / "scaler_params.json", "r") as f:
    scp = json.load(f)
scaler_mean = np.array(scp["mean"], dtype=np.float32)
scaler_std = np.array(scp["std"], dtype=np.float32)

with open(script_dir / "Motion Library" / "cluster_centroids.json", "r") as f:
    CLUSTER_CENTROIDS = np.array(json.load(f), dtype=np.float32)

# Participant-specific cluster assignment
participant_clusters = None
seed = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(),16) & 0xFFFFFFFF
rng = np.random.default_rng(seed)

# -------------------------------------------------------------------
#  Helpers for simulation mode
# -------------------------------------------------------------------
class SimulatedMouse:
    def __init__(self):
        self._pos = np.array([0.0, 0.0], dtype=float)
    def setPos(self, pos=(0, 0)):
        self._pos = np.array(pos, dtype=float)
    def getPos(self):
        self._pos += rng.normal(0, 3, 2)
        return self._pos.tolist()

def wait_keys(keys=None):
    if SIMULATE:
        if keys is None:
            core.wait(0.2)
            return ["space"]
        allowed = [k for k in keys if k != "escape"] or ["space"]
        return [rng.choice(allowed)]
    return event.waitKeys(keyList=keys)

# ───────────────────────────────────────────────────────
#  Trajectory Quality Control and Preprocessing
# ───────────────────────────────────────────────────────

def analyze_trajectory_quality(trajectory):
    """Analyze trajectory quality metrics to identify problematic patterns"""
    # Calculate frame-by-frame velocities
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Quality metrics
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    max_speed = np.max(speeds)
    min_speed = np.min(speeds)
    
    # Movement consistency
    zero_movement_ratio = np.sum(speeds < 0.5) / len(speeds)  # ratio of near-zero movement
    high_jitter_ratio = np.sum(speeds > mean_speed + 3*std_speed) / len(speeds)  # outlier spikes
    
    # Path smoothness (angle changes)
    if len(velocities) > 1:
        unit_velocities = velocities / (speeds.reshape(-1, 1) + 1e-9)
        angle_changes = np.arccos(np.clip(np.sum(unit_velocities[:-1] * unit_velocities[1:], axis=1), -1, 1))
        mean_angle_change = np.mean(angle_changes)
        jerkiness = np.std(angle_changes)
    else:
        mean_angle_change = 0
        jerkiness = 0
    
    return {
        'mean_speed': mean_speed,
        'std_speed': std_speed,
        'zero_movement_ratio': zero_movement_ratio,
        'high_jitter_ratio': high_jitter_ratio,
        'mean_angle_change': mean_angle_change,
        'jerkiness': jerkiness,
        'speed_range': max_speed - min_speed
    }

def is_trajectory_valid(trajectory, min_speed=1.0, max_zero_ratio=0.3, max_jitter_ratio=0.1, max_jerkiness=1.5):
    """Check if trajectory meets quality criteria"""
    quality = analyze_trajectory_quality(trajectory)
    
    # Quality checks
    if quality['mean_speed'] < min_speed:
        return False, "mean_speed_too_low"
    if quality['zero_movement_ratio'] > max_zero_ratio:
        return False, "too_much_zero_movement"
    if quality['high_jitter_ratio'] > max_jitter_ratio:
        return False, "too_much_jitter"
    if quality['jerkiness'] > max_jerkiness:
        return False, "too_jerky"
    
    return True, "valid"

def normalize_trajectory(trajectory, target_speed_range=(2.0, 15.0), smooth_factor=0.7):
    """Normalize trajectory to target speed range and apply smoothing"""
    if len(trajectory) < 2:
        return trajectory
    
    # Calculate velocities and speeds
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Normalize to target speed range
    current_mean_speed = np.mean(speeds)
    if current_mean_speed > 0:
        target_mean_speed = np.mean(target_speed_range)
        speed_scale = target_mean_speed / current_mean_speed
        velocities = velocities * speed_scale
    
    # Apply smoothing
    smoothed_velocities = velocities.copy()
    for i in range(1, len(velocities)):
        smoothed_velocities[i] = smooth_factor * smoothed_velocities[i-1] + (1 - smooth_factor) * velocities[i]
    
    # Reconstruct trajectory
    normalized_trajectory = [trajectory[0]]
    for vel in smoothed_velocities:
        next_point = normalized_trajectory[-1] + vel
        normalized_trajectory.append(next_point)
    
    return np.array(normalized_trajectory)

def preprocess_motion_pool():
    """Preprocess motion pool to ensure quality and consistency"""
    global motion_pool, snippet_features, snippet_labels
    
    print("Preprocessing motion pool for quality control...")
    
    # Track valid snippets
    valid_indices = []
    processed_snippets = []
    processed_features = []
    processed_labels = []
    
    for i, snippet in enumerate(motion_pool):
        # Convert snippet to trajectory format for analysis
        trajectory = np.cumsum(snippet, axis=0)
        
        # Check quality
        is_valid, reason = is_trajectory_valid(trajectory)
        
        if is_valid:
            # Normalize if needed
            normalized_trajectory = normalize_trajectory(trajectory)
            velocities = np.diff(normalized_trajectory, axis=0)
            
            processed_snippets.append(velocities)
            processed_features.append(snippet_features[i])
            processed_labels.append(snippet_labels[i])
            valid_indices.append(i)
        else:
            print(f"Removed snippet {i}: {reason}")
    
    # Update global variables
    motion_pool = np.array(processed_snippets)
    snippet_features = np.array(processed_features)
    snippet_labels = np.array(processed_labels)
    
    print(f"Motion pool preprocessed: {len(valid_indices)}/{len(motion_pool)} snippets retained")
    return valid_indices

# Preprocess motion pool
valid_snippet_indices = preprocess_motion_pool()

def get_trajectory_signature(trajectory):
    """Extract movement signature features from trajectory"""
    if len(trajectory) < 2:
        return np.zeros(6)
    
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Basic features
    mean_speed = np.mean(speeds)
    std_speed = np.std(speeds)
    
    # Directional features
    if len(velocities) > 1:
        unit_velocities = velocities / (speeds.reshape(-1, 1) + 1e-9)
        dot_products = np.sum(unit_velocities[:-1] * unit_velocities[1:], axis=1)
        dot_products = np.clip(dot_products, -1, 1)
        angles = np.arccos(dot_products)
        mean_turn = np.mean(angles)
        std_turn = np.std(angles)
    else:
        mean_turn = std_turn = 0
    
    # Path efficiency
    net_displacement = trajectory[-1] - trajectory[0]
    path_length = np.sum(speeds)
    straightness = np.linalg.norm(net_displacement) / (path_length + 1e-9)
    
    return np.array([mean_speed, std_speed, mean_turn, std_turn, straightness, 0])  # 6D feature space

def find_matched_trajectory_pair():
    """Find a pair of trajectories with similar movement signatures"""
    if len(valid_snippet_indices) < 2:
        return None, None
    
    # Get participant's movement signature from demo trial
    # For now, use a placeholder - this will be updated after demo trial
    participant_signature = np.array([5.0, 2.0, 0.5, 0.3, 0.7, 0.0])  # Placeholder
    
    # Normalize signature using same parameters as training
    normalized_signature = (participant_signature - scaler_mean) / (scaler_std + 1e-9)
    
    # Find closest matches in valid snippets
    distances = []
    for idx in valid_snippet_indices:
        snippet_feat = snippet_features[idx]
        distance = np.linalg.norm(normalized_signature - snippet_feat)
        distances.append((distance, idx))
    
    # Sort by distance and select two different snippets
    distances.sort()
    
    if len(distances) >= 2:
        return distances[0][1], distances[1][1]
    else:
        return None, None

def apply_consistent_smoothing(trajectory1, trajectory2):
    """Apply consistent smoothing to both trajectories"""
    def smooth_trajectory(traj, window_size=5):
        if len(traj) < window_size:
            return traj
        
        smoothed = traj.copy()
        for i in range(len(traj)):
            start = max(0, i - window_size // 2)
            end = min(len(traj), i + window_size // 2 + 1)
            smoothed[i] = np.mean(traj[start:end], axis=0)
        
        return smoothed
    
    # Convert velocity snippets to position trajectories
    pos1 = np.cumsum(trajectory1, axis=0)
    pos2 = np.cumsum(trajectory2, axis=0)
    
    # Apply smoothing
    smooth_pos1 = smooth_trajectory(pos1)
    smooth_pos2 = smooth_trajectory(pos2)
    
    # Convert back to velocities
    vel1 = np.diff(smooth_pos1, axis=0)
    vel2 = np.diff(smooth_pos2, axis=0)
    
    return vel1, vel2

def assign_participant_clusters(demo_mean_speed, demo_mean_turn, demo_straightness):
    """Assign participant to their 2 closest clusters based on demo trial features"""
    # Create participant feature vector (6D, with 3 zeros for missing features)
    participant_features = np.array([demo_mean_speed, demo_mean_turn, demo_straightness, 0, 0, 0], dtype=np.float32)
    
    # Normalize using same parameters as training
    normalized_features = (participant_features - scaler_mean) / (scaler_std + 1e-9)
    
    # Calculate distances to all cluster centroids
    distances = []
    for i, centroid in enumerate(CLUSTER_CENTROIDS):
        distance = np.linalg.norm(normalized_features - centroid)
        distances.append((distance, i))
    
    # Sort by distance and select 2 closest clusters
    distances.sort()
    closest_clusters = [distances[0][1], distances[1][1]]
    
    print(f"Participant features: speed={demo_mean_speed:.2f}, turn={demo_mean_turn:.2f}, straightness={demo_straightness:.2f}")
    print(f"Assigned to clusters: {closest_clusters}")
    
    return closest_clusters

def sample_from_participant_clusters(n_samples_per_cluster=10):
    """Sample snippets from participant's assigned clusters"""
    if participant_clusters is None:
        print("Warning: No participant clusters assigned, using random sampling")
        return rng.choice(valid_snippet_indices, size=2*n_samples_per_cluster, replace=False)
    
    # Get snippets from participant's clusters
    cluster_snippets = {cluster: [] for cluster in participant_clusters}
    
    for idx in valid_snippet_indices:
        if snippet_labels[idx] in participant_clusters:
            cluster_snippets[snippet_labels[idx]].append(idx)
    
    # Sample from each cluster
    selected_snippets = []
    for cluster in participant_clusters:
        cluster_indices = cluster_snippets[cluster]
        if len(cluster_indices) >= n_samples_per_cluster:
            selected = rng.choice(cluster_indices, size=n_samples_per_cluster, replace=False)
        else:
            # If not enough snippets in cluster, use all available
            selected = cluster_indices
            print(f"Warning: Only {len(cluster_indices)} snippets available in cluster {cluster}")
        
        selected_snippets.extend(selected)
    
    print(f"Sampled {len(selected_snippets)} snippets from participant clusters {participant_clusters}")
    return selected_snippets

# ───────────────────────────────────────────────────────
#  Constants and Parameters
# ───────────────────────────────────────────────────────
OFFSET_X = 300
LOWPASS = 0.8

# ───────────────────────────────────────────────────────
#  Paths & ExperimentHandler
# ───────────────────────────────────────────────────────
# Always save data in the Main Experiment/data directory, regardless of CWD
root = pathlib.Path(__file__).parent / "data"; root.mkdir(exist_ok=True)

participant_id = expInfo['participant']
base_filename = f"CDT_v2_{participant_id}"
csv_path = root / f"{base_filename}.csv"
kinematics_csv_path = root / f"{base_filename}_kinematics.csv"

# Prevent overwriting data
i = 1
while csv_path.exists():
    new_filename = f"CDT_v2_{participant_id}_{i}"
    csv_path = root / f"{new_filename}.csv"
    kinematics_csv_path = root / f"{new_filename}_kinematics.csv"
    i += 1

thisExp = data.ExperimentHandler(
    name=expName, extraInfo=expInfo,
    savePickle=False, saveWideText=False,
    dataFileName=str(root / base_filename)
)

# ───────────────────────────────────────────────────────
#  Window & stimuli
# ───────────────────────────────────────────────────────
win = visual.Window((1920,1080), fullscr=not SIMULATE, color=[0.5]*3, units="pix", allowGUI=True)
win.setMouseVisible(False)
square = visual.Rect(win, 40, 40, fillColor="black", lineColor="black")
dot = visual.Circle(win, 20, fillColor="black", lineColor="black")
fix = visual.TextStim(win, "+", color="white", height=60)
msg = visual.TextStim(win, "", color="white", height=26, wrapWidth=1000)
feedbackTxt = visual.TextStim(win, "", color="black", height=80)

confine = lambda p, l=250: p if (r := math.hypot(*p)) <= l else (p[0]*l/r, p[1]*l/r)
rotate = lambda vx, vy, a: (
    vx * math.cos(math.radians(a)) - vy * math.sin(math.radians(a)),
    vx * math.sin(math.radians(a)) + vy * math.cos(math.radians(a))
)

# ───────────────────────────────────────────────────────
#  Demo Trial
# ───────────────────────────────────────────────────────
def demo():
    global avg_demo_speed, participant_clusters
    prop_demo = 0.80  # High control for demo
    
    msg.text = "DEMO TRIAL – Get familiar with the task.\nPress any key."
    msg.draw(); win.flip(); wait_keys()
    
    # Use same trial structure as main trials
    # Use a simple color for demo (not the expectation colors)
    demo_color = "white"  # Use white for demo to avoid confusion
    fix.color = demo_color; square.fillColor = square.lineColor = demo_color; dot.fillColor = dot.lineColor = demo_color
    
    fix.draw(); win.flip(); core.wait(1.0)
    
    # Initial stimulus presentation (identical to main trials)
    left_shape = random.choice(['square', 'dot'])
    if left_shape == 'square':
        square.pos = (-OFFSET_X, 0)
        dot.pos = (OFFSET_X, 0)
    else:
        square.pos = (OFFSET_X, 0)
        dot.pos = (-OFFSET_X, 0)
    
    square.draw(); dot.draw(); win.flip()
    
    mouse = SimulatedMouse() if SIMULATE else event.Mouse(win=win, visible=False)
    mouse.setPos((0, 0))
    last = mouse.getPos()
    while True:
        square.draw(); dot.draw(); win.flip()
        x, y = mouse.getPos()
        if math.hypot(x - last[0], y - last[1]) > 0 or SIMULATE: break
        if not SIMULATE and event.getKeys(["escape"]): _save(); core.quit()
    
    # Get matched trajectory pair (identical to main trials)
    target_snippet_idx, distractor_snippet_idx = find_matched_trajectory_pair()
    
    # Fallback to random valid snippets if matching fails
    if target_snippet_idx is None or distractor_snippet_idx is None:
        print("Demo: Trajectory matching failed, using fallback...")
        if len(valid_snippet_indices) >= 2:
            selected = rng.choice(valid_snippet_indices, size=2, replace=False)
            target_snippet_idx, distractor_snippet_idx = selected[0], selected[1]
        else:
            print("Demo: No valid snippets available, using random...")
            target_snippet_idx = distractor_snippet_idx = rng.integers(0, TOTAL_SNIPS)
    
    target_snippet = motion_pool[target_snippet_idx]
    distractor_snippet = motion_pool[distractor_snippet_idx]
    
    # Apply consistent smoothing (identical to main trials)
    try:
        target_snippet, distractor_snippet = apply_consistent_smoothing(target_snippet, distractor_snippet)
    except Exception as e:
        print(f"Demo: Smoothing failed, using original trajectories: {e}")
        # Use original trajectories if smoothing fails
    
    demo_positions = []; speeds = []
    clk = core.Clock(); frame = 0
    vt = vd = np.zeros(2, np.float32)
    
    # Main trial loop (identical to main trials)
    while clk.getTime() < 3.0:
        x, y = mouse.getPos()
        demo_positions.append((x, y))
        dx, dy = x - last[0], y - last[1]
        last = (x, y); speeds.append(math.hypot(dx, dy))
        
        # Get velocities from both snippets (identical to main trials)
        target_ou_dx, target_ou_dy = target_snippet[frame % SNIP_LEN]
        distractor_ou_dx, distractor_ou_dy = distractor_snippet[frame % SNIP_LEN]
        frame += 1
        
        # Normalize snippets to mouse movement magnitude (identical to main trials)
        mag_m = math.hypot(dx, dy)
        mag_target = math.hypot(target_ou_dx, target_ou_dy)
        if mag_target > 0: 
            target_ou_dx, target_ou_dy = target_ou_dx / mag_target * mag_m, target_ou_dy / mag_target * mag_m
        
        mag_distractor = math.hypot(distractor_ou_dx, distractor_ou_dy)
        if mag_distractor > 0:
            distractor_ou_dx, distractor_ou_dy = distractor_ou_dx / mag_distractor * mag_m, distractor_ou_dy / mag_distractor * mag_m
        
        # Mix mouse and trajectory movements (identical to main trials)
        tdx = prop_demo * dx + (1 - prop_demo) * target_ou_dx
        tdy = prop_demo * dy + (1 - prop_demo) * target_ou_dy
        ddx = distractor_ou_dx  # Distractor uses only its trajectory
        ddy = distractor_ou_dy
        
        vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
        vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ddx, ddy])
        
        # Move shapes (identical to main trials)
        square.pos = confine(tuple(square.pos + vt))
        dot.pos = confine(tuple(dot.pos + vd))
        square.draw(); dot.draw(); win.flip()
    
    # Freeze shapes in final position (identical to main trials)
    square.draw(); dot.draw(); win.flip(); core.wait(0.5)
    
    avg_demo_speed = float(np.mean(speeds))
    demo_arr = np.array(demo_positions, dtype=np.float32)
    deltas = np.diff(demo_arr, axis=0); demo_speeds = np.linalg.norm(deltas, axis=1)
    demo_mean_speed = float(np.mean(demo_speeds))
    unit = deltas / (demo_speeds.reshape(-1, 1) + 1e-9)
    dotp = np.einsum('ij,ij->i', unit[:-1], unit[1:])
    dotp = np.clip(dotp, -1.0, 1.0); angles = np.arccos(dotp)
    demo_mean_turn = float(np.mean(angles))
    net_disp = demo_arr[-1] - demo_arr[0]; net_mag = np.linalg.norm(net_disp)
    path_len = np.sum(demo_speeds)
    demo_straightness = float(net_mag / (path_len + 1e-9))
    
    # Assign participant to their 2 closest clusters
    participant_clusters = assign_participant_clusters(demo_mean_speed, demo_mean_turn, demo_straightness)
    print(f"Participant assigned to clusters {participant_clusters}")

# ───────────────────────────────────────────────────────
#  Staircase and Trial Parameters
# ───────────────────────────────────────────────────────
import numpy as np

def accuracy_to_logit(accuracy):
    """Convert accuracy to logit space"""
    # Avoid infinite values
    accuracy = np.clip(accuracy, 0.01, 0.99)
    return np.log(accuracy / (1 - accuracy))

def logit_to_accuracy(logit):
    """Convert logit back to accuracy"""
    return 1 / (1 + np.exp(-logit))

def prop_to_logit_midpoint(prop_low, prop_high):
    """Calculate logit-space midpoint between two control proportions
    
    This assumes the proportions correspond to accuracy levels that should
    be averaged in logit space to find the perceptual midpoint.
    """
    # For now, we'll assume the proportions roughly correspond to accuracy
    # In a more sophisticated approach, you could empirically map prop -> accuracy
    
    # Rough mapping: higher prop = higher accuracy (this is task-dependent)
    # We'll use the proportions directly as a proxy for accuracy
    logit_low = accuracy_to_logit(prop_low)
    logit_high = accuracy_to_logit(prop_high) 
    logit_mid = (logit_low + logit_high) / 2
    return logit_to_accuracy(logit_mid)

MODES = [0, 90]; EXPECT = ["low", "high"]
TARGET = {"low": 0.85, "high": 0.55}; BASE = 0.05
STEP_DOWN = {e: BASE for e in EXPECT}
STEP_UP = {e: (TARGET[e]/(1-TARGET[e]))*BASE for e in EXPECT}
PROP = {m: {"low": 0.30, "high": 0.60} for m in MODES}
MED_PROP = {m: (PROP[m]["low"]+PROP[m]["high"])/2 for m in MODES}
TEST_PROP = MED_PROP.copy()

# ───────────────────────────────────────────────────────
#  Main Experiment Loop
# ───────────────────────────────────────────────────────

# Run demo trial
demo()

# Main experiment trials
for trial_num in range(1, 21):  # 20 trials
    # Randomize trial parameters
    mode = random.choice(MODES)
    expect_level = random.choice(EXPECT)
    
    # Run trial (simplified for this example)
    print(f"Trial {trial_num}: Mode={mode}, Expect={expect_level}")
    
    # Here you would call your run_trial function
    # run_trial(trial_num, "test", 0, expect_level, mode)

print("Experiment completed!")
win.close()
core.quit() 