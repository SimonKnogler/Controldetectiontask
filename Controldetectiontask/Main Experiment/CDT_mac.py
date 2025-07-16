#!/opt/homebrew/bin/python3
# -*- coding: utf-8 -*-
"""
control_detection_task_v2.py - 4-cluster version with participant-specific sampling
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
    # If psychopy import fails, run with anaconda Python
    try:
        import numpy as np
        import pandas as pd
        from psychopy import visual, event, core, data, gui
        return False  # Continue with current interpreter
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Switching to anaconda Python with all required packages...")
        anaconda_python = "/opt/anaconda3/bin/python"
        if os.path.exists(anaconda_python):
            # Re-run this script with anaconda Python
            result = subprocess.run([anaconda_python] + sys.argv, check=False)
            sys.exit(result.returncode)
        else:
            print("Error: Anaconda Python not found. Please install required packages.")
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
LIB_NAME = "/Users/simonknogler/Desktop/PhD/WP1/Controldetectiontask/Motion Library/core_pool.npy"
FEATS_NAME = "/Users/simonknogler/Desktop/PhD/WP1/Controldetectiontask/Motion Library/core_pool_feats.npy"
LABELS_NAME = "/Users/simonknogler/Desktop/PhD/WP1/Controldetectiontask/Motion Library/core_pool_labels.npy"

motion_pool = np.load(LIB_NAME)
snippet_features = np.load(FEATS_NAME)
snippet_labels = np.load(LABELS_NAME)

SNIP_LEN = motion_pool.shape[1]
TOTAL_SNIPS = motion_pool.shape[0]
K_CLUST = 4

print(f"Loaded {TOTAL_SNIPS} snippets × {SNIP_LEN} frames from {LIB_NAME}")
print(f"Cluster distribution: {np.bincount(snippet_labels)}")

# Load scaler parameters and cluster centroids
with open("/Users/simonknogler/Desktop/PhD/WP1/Controldetectiontask/Motion Library/scaler_params.json", "r") as f:
    scp = json.load(f)
scaler_mean = np.array(scp["mean"], dtype=np.float32)
scaler_std = np.array(scp["scale"], dtype=np.float32)  # Use "scale" instead of "std"

with open("/Users/simonknogler/Desktop/PhD/WP1/Controldetectiontask/Motion Library/cluster_centroids.json", "r") as f:
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
    """Normalize trajectory to ensure consistent movement characteristics"""
    # Calculate current velocities
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Avoid division by zero
    speeds = np.maximum(speeds, 0.1)
    
    # Normalize speed to target range
    current_speed_range = (np.percentile(speeds, 10), np.percentile(speeds, 90))
    if current_speed_range[1] > current_speed_range[0]:
        # Map current speed range to target range
        speed_scale = (target_speed_range[1] - target_speed_range[0]) / (current_speed_range[1] - current_speed_range[0])
        normalized_speeds = (speeds - current_speed_range[0]) * speed_scale + target_speed_range[0]
    else:
        normalized_speeds = np.full_like(speeds, np.mean(target_speed_range))
    
    # Apply smoothing to reduce jitter
    for i in range(1, len(normalized_speeds)):
        normalized_speeds[i] = smooth_factor * normalized_speeds[i-1] + (1-smooth_factor) * normalized_speeds[i]
    
    # Reconstruct trajectory with normalized speeds
    normalized_velocities = velocities / speeds.reshape(-1, 1) * normalized_speeds.reshape(-1, 1)
    normalized_trajectory = np.zeros_like(trajectory)
    normalized_trajectory[0] = trajectory[0]  # Keep starting position
    
    # Rebuild positions from normalized velocities
    for i in range(1, len(trajectory)):
        normalized_trajectory[i] = normalized_trajectory[i-1] + normalized_velocities[i-1]
    
    # Center the trajectory
    normalized_trajectory = normalized_trajectory - np.mean(normalized_trajectory, axis=0)
    
    return normalized_trajectory

def preprocess_motion_pool():
    """Filter and preprocess the motion pool to remove problematic trajectories"""
    global motion_pool, snippet_labels, valid_snippet_indices
    
    print("Preprocessing motion pool...")
    valid_indices = []
    invalid_count = 0
    invalid_reasons = {}
    
    for i in range(TOTAL_SNIPS):
        trajectory = motion_pool[i]  # Shape: (SNIP_LEN, 2)
        
        # Check trajectory quality
        is_valid, reason = is_trajectory_valid(trajectory)
        
        if is_valid:
            # Normalize the trajectory
            motion_pool[i] = normalize_trajectory(trajectory)
            valid_indices.append(i)
        else:
            invalid_count += 1
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    
    valid_snippet_indices = np.array(valid_indices)
    
    print(f"Motion pool preprocessing complete:")
    print(f"  Valid trajectories: {len(valid_indices)}/{TOTAL_SNIPS}")
    print(f"  Filtered out: {invalid_count}")
    if invalid_reasons:
        for reason, count in invalid_reasons.items():
            print(f"    {reason}: {count}")
    
    return valid_snippet_indices

# ───────────────────────────────────────────────────────
#  Enhanced Trajectory Matching for Target/Distractor Pairing
# ───────────────────────────────────────────────────────

def get_trajectory_signature(trajectory):
    """Get a signature that describes key movement characteristics"""
    velocities = np.diff(trajectory, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)
    
    # Key characteristics for matching
    return {
        'mean_speed': np.mean(speeds),
        'speed_variability': np.std(speeds),
        'path_length': np.sum(speeds),
        'net_displacement': np.linalg.norm(trajectory[-1] - trajectory[0]),
        'speed_percentiles': np.percentile(speeds, [25, 50, 75])
    }

def find_matched_trajectory_pair():
    """Find two trajectories with similar movement characteristics for target and distractor"""
    global valid_snippet_indices
    
    if len(valid_snippet_indices) < 2:
        return None, None
    
    # For efficiency, sample a subset of valid trajectories
    sample_size = min(100, len(valid_snippet_indices))
    candidate_indices = rng.choice(valid_snippet_indices, size=sample_size, replace=False)
    
    # Get signatures for all candidates
    signatures = []
    for idx in candidate_indices:
        trajectory = motion_pool[idx]
        sig = get_trajectory_signature(trajectory)
        signatures.append((idx, sig))
    
    # Find the best matching pair
    best_score = float('inf')
    best_pair = (None, None)
    
    for i in range(len(signatures)):
        for j in range(i + 1, len(signatures)):
            idx1, sig1 = signatures[i]
            idx2, sig2 = signatures[j]
            
            # Calculate similarity score (lower is better)
            speed_diff = abs(sig1['mean_speed'] - sig2['mean_speed'])
            var_diff = abs(sig1['speed_variability'] - sig2['speed_variability'])
            length_diff = abs(sig1['path_length'] - sig2['path_length']) / max(sig1['path_length'], sig2['path_length'])
            
            # Combined similarity score
            similarity_score = speed_diff + var_diff + length_diff * 10
            
            if similarity_score < best_score:
                best_score = similarity_score
                best_pair = (idx1, idx2)
    
    return best_pair

def apply_consistent_smoothing(trajectory1, trajectory2):
    """Apply consistent smoothing to both trajectories to reduce any remaining artifacts"""
    def smooth_trajectory(traj, window_size=5):
        """Apply simple moving average smoothing"""
        smoothed = np.copy(traj)
        for i in range(window_size, len(traj) - window_size):
            smoothed[i] = np.mean(traj[i-window_size:i+window_size], axis=0)
        return smoothed
    
    return smooth_trajectory(trajectory1), smooth_trajectory(trajectory2)

# Initialize preprocessing
valid_snippet_indices = preprocess_motion_pool()
print(f"Using {len(valid_snippet_indices)} valid trajectories out of {TOTAL_SNIPS} total")

# ───────────────────────────────────────────────────────
#  Helper: assign participant to 2 closest clusters
# ───────────────────────────────────────────────────────
def assign_participant_clusters(demo_mean_speed, demo_mean_turn, demo_straightness):
    demo_vector = np.array([
        demo_mean_speed, 0.0, demo_mean_turn, 0.0, demo_straightness, 0.0
    ], dtype=np.float32)
    demo_scaled = (demo_vector - scaler_mean) / (scaler_std + 1e-9)
    dists = np.linalg.norm(CLUSTER_CENTROIDS - demo_scaled.reshape(1, -1), axis=1)
    
    # Get indices of 2 closest clusters
    closest_indices = np.argsort(dists)[:2]
    return closest_indices.tolist()

# ───────────────────────────────────────────────────────
#  Helper: sample snippets from participant's assigned clusters
# ───────────────────────────────────────────────────────
def sample_from_participant_clusters(n_samples_per_cluster=10):
    global participant_clusters, valid_snippet_indices
    
    if participant_clusters is None:
        print("Error: Participant clusters not assigned yet!")
        return []
    
    selected_snippets = []
    for cluster_id in participant_clusters:
        # Find all valid snippets from this cluster
        cluster_mask = snippet_labels == cluster_id
        cluster_snippets = np.where(cluster_mask)[0]
        
        # Filter to only include valid snippets
        valid_cluster_snippets = np.intersect1d(cluster_snippets, valid_snippet_indices)
        
        # Sample without replacement from valid snippets
        n_available = len(valid_cluster_snippets)
        n_to_sample = min(n_samples_per_cluster, n_available)
        
        if n_to_sample > 0:
            selected_indices = rng.choice(valid_cluster_snippets, size=n_to_sample, replace=False)
            selected_snippets.extend(selected_indices.tolist())
        else:
            print(f"Warning: No valid snippets available for cluster {cluster_id}")
    
    return selected_snippets

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
BREAK_EVERY = 15; LOWPASS = 0.8; OFFSET_X = 150
CONF_KEYS = ["1", "2", "3", "4"]
SHAPE_FROM_KEY = {"1": "square", "2": "square", "3": "dot", "4": "dot"}
CONF_FROM_KEY = {k: i+1 for i, k in enumerate(CONF_KEYS)}

demo()

# ───────────────────────────────────────────────────────
#  Trial function
# ───────────────────────────────────────────────────────
def run_trial(trial_num, phase, angle_bias, expect_level, mode, catch_type="", target_shape=None):
    if catch_type == "full": 
        prop = 1.0
    elif mode == "true": 
        prop = PROP[angle_bias][expect_level]
    else: 
        prop = TEST_PROP[angle_bias]

    cue = low_col if expect_level == "low" else high_col
    fix.color = cue; square.fillColor = square.lineColor = cue; dot.fillColor = dot.lineColor = cue

    fix.draw(); win.flip(); core.wait(1.0)

    # Initial stimulus presentation
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

    target = target_shape if target_shape is not None else random.choice(["square", "dot"])
    
    # Get matched trajectory pair for target and distractor
    target_snippet_idx, distractor_snippet_idx = find_matched_trajectory_pair()
    
    # Fallback to participant clusters if matching fails
    if target_snippet_idx is None or distractor_snippet_idx is None:
        available_snippets = sample_from_participant_clusters(n_samples_per_cluster=20)
        if not available_snippets:
            # Final fallback to random valid snippets
            if len(valid_snippet_indices) >= 2:
                selected = rng.choice(valid_snippet_indices, size=2, replace=False)
                target_snippet_idx, distractor_snippet_idx = selected[0], selected[1]
            else:
                print("Error: Insufficient valid snippets!")
                target_snippet_idx = distractor_snippet_idx = rng.integers(0, TOTAL_SNIPS)
        else:
            # Use two different snippets from available ones
            if len(available_snippets) >= 2:
                selected = rng.choice(available_snippets, size=2, replace=False)
                target_snippet_idx, distractor_snippet_idx = selected[0], selected[1]
            else:
                target_snippet_idx = distractor_snippet_idx = rng.choice(available_snippets)
    
    target_snippet = motion_pool[target_snippet_idx]
    distractor_snippet = motion_pool[distractor_snippet_idx]
    
    # Apply consistent smoothing to both trajectories
    target_snippet, distractor_snippet = apply_consistent_smoothing(target_snippet, distractor_snippet)
    
    cluster_id = snippet_labels[target_snippet_idx]  # Use target's cluster for logging

    trial_kinematics = []
    clk = core.Clock(); frame = 0
    vt = vd = np.zeros(2, np.float32)

    while clk.getTime() < 3.0:
        x, y = mouse.getPos()
        dx, dy = x - last[0], y - last[1]
        last = (x, y)
        dx, dy = rotate(dx, dy, angle_bias)
        
        # Get velocities from both snippets
        target_ou_dx, target_ou_dy = target_snippet[frame % SNIP_LEN]
        distractor_ou_dx, distractor_ou_dy = distractor_snippet[frame % SNIP_LEN]
        frame += 1
        
        # Normalize target snippet to mouse movement magnitude
        mag_m = math.hypot(dx, dy)
        mag_target = math.hypot(target_ou_dx, target_ou_dy)
        if mag_target > 0: 
            target_ou_dx, target_ou_dy = target_ou_dx / mag_target * mag_m, target_ou_dy / mag_target * mag_m
        
        # Normalize distractor snippet to mouse movement magnitude  
        mag_distractor = math.hypot(distractor_ou_dx, distractor_ou_dy)
        if mag_distractor > 0:
            distractor_ou_dx, distractor_ou_dy = distractor_ou_dx / mag_distractor * mag_m, distractor_ou_dy / mag_distractor * mag_m
        
        # Mix mouse and trajectory movements
        tdx = prop * dx + (1 - prop) * target_ou_dx
        tdy = prop * dy + (1 - prop) * target_ou_dy
        ddx = distractor_ou_dx  # Distractor uses only its trajectory
        ddy = distractor_ou_dy
        
        vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
        vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ddx, ddy])

        if target == "square":
            square.pos = confine(tuple(square.pos + vt))
            dot.pos = confine(tuple(dot.pos + vd))
        else:
            dot.pos = confine(tuple(dot.pos + vt))
            square.pos = confine(tuple(square.pos + vd))

        trial_kinematics.append({
            'timestamp': clk.getTime(), 'frame': frame, 'mouse_x': x, 'mouse_y': y,
            'square_x': square.pos[0], 'square_y': square.pos[1], 'dot_x': dot.pos[0], 'dot_y': dot.pos[1],
            'cluster_id': cluster_id
        })
        square.draw(); dot.draw(); win.flip()

    # Freeze shapes in final position for 500ms
    square.draw(); dot.draw(); win.flip(); core.wait(0.5)
    
    msg.text = ("Which shape did you control & how confident?\n1 Sq(guess) 2 Sq(conf) 3 Dot(conf) 4 Dot(guess)")
    msg.draw(); win.flip()
    t0 = core.getTime()
    key = wait_keys(CONF_KEYS + ["escape"])[0]
    rt_choice = core.getTime() - t0
    if key == "escape": _save(); core.quit()
    resp_shape = SHAPE_FROM_KEY[key]
    conf_lvl = CONF_FROM_KEY[key]
    correct = int(resp_shape == target)

    if phase == "practice":
        # Professional feedback based on agency research standards
        if correct:
            feedbackTxt.text = "Right"
        else:
            feedbackTxt.text = "Wrong"
        
        feedbackTxt.draw(); win.flip(); core.wait(0.8)
        win.flip(); core.wait(0.3)

    # Clear any buffered key presses before agency rating
    if phase == "test":
        event.clearEvents(eventType='keyboard')
    
    rating = np.nan
    if phase == "test":
        if SIMULATE: 
            rating = float(rng.integers(0, 101))
        else:
            # Agency rating scale (based on standard agency research protocols)
            # Show keyboard instruction only on first test trial (after practice block)
            if phase == "test" and trial_counter == len(practice) + 1:
                msg.text = "How much control did you feel over the shape's movement?\n\n(Choose with the numbers 1-7 on your keyboard)"
            else:
                msg.text = "How much control did you feel over the shape's movement?"
            
            # Create horizontal rating scale display
            scale_positions = [(-450, -100), (-300, -100), (-150, -100), (0, -100), (150, -100), (300, -100), (450, -100)]
            scale_labels = [
                "1\nVery weak",
                "2\nWeak", 
                "3\nSomewhat weak",
                "4\nModerate",
                "5\nSomewhat strong",
                "6\nStrong",
                "7\nVery strong"
            ]
            
            # Create individual text stimuli for each scale point
            scale_stimuli = []
            for i, (pos, label) in enumerate(zip(scale_positions, scale_labels)):
                scale_stim = visual.TextStim(win, text=label, pos=pos, height=18, color='white', alignText='center')
                scale_stimuli.append(scale_stim)
            
            rating = None
            while rating is None:
                # Display question and scale immediately
                msg.draw()
                for stim in scale_stimuli:
                    stim.draw()
                win.flip()
                
                # Check for key presses
                keys = event.getKeys(['1', '2', '3', '4', '5', '6', '7', 'escape'])
                if keys:
                    if 'escape' in keys:
                        _save(); core.quit()
                    else:
                        # Convert key to rating (1-7 scale)
                        rating = int(keys[0])
                
                core.wait(0.01)
            
            core.wait(0.2)  # Hide mouse again after rating

    for frame_data in trial_kinematics:
        frame_data.update({
            'participant': expInfo['participant'], 'session': expInfo['session'], 
            'trial_num': trial_num, 'phase': phase, 'angle_bias': angle_bias, 
            'expect_level': expect_level, 'prop_used': prop, 'conf_level': conf_lvl, 
            'agency_rating': rating
        })
        kinematics_data.append(frame_data)

    if phase == "practice" and catch_type == "":
        if correct: 
            PROP[angle_bias][expect_level] = max(0, PROP[angle_bias][expect_level] - STEP_DOWN[expect_level])
        else: 
            PROP[angle_bias][expect_level] = min(1, PROP[angle_bias][expect_level] + STEP_UP[expect_level])

    return dict(
        target_snippet_id=target_snippet_idx, distractor_snippet_id=distractor_snippet_idx, 
        cluster_id=cluster_id, catch_type=catch_type, phase=phase, 
        angle_bias=angle_bias, expect_level=expect_level, true_shape=target, resp_shape=resp_shape,
        conf_level=conf_lvl, accuracy=correct, rt_choice=rt_choice, agency_rating=rating, prop_used=prop
    )

# ───────────────────────────────────────────────────────
#  Practice block
# ───────────────────────────────────────────────────────
PPC = 20
practice = [(m, e) for m in MODES for e in EXPECT] * PPC
random.shuffle(practice)
msg.text = "PRACTICE – Press any key."; msg.draw(); win.flip(); wait_keys()
trial_counter = 0
for m, e in practice:
    trial_counter += 1
    res = run_trial(trial_counter, "practice", m, e, "true")
    for k, v in res.items(): thisExp.addData(k, v)
    thisExp.nextEntry()

# Calculate logit-space midpoint for test trials (after practice adaptation)
MED_PROP = {}
for m in MODES:
    # Use logit-space averaging for psychometrically correct midpoint
    logit_midpoint = prop_to_logit_midpoint(PROP[m]["low"], PROP[m]["high"])
    MED_PROP[m] = logit_midpoint
    print(f"Mode {m}°: Low={PROP[m]['low']:.3f}, High={PROP[m]['high']:.3f}, Logit-midpoint={logit_midpoint:.3f}")

TEST_PROP = MED_PROP.copy()

# ───────────────────────────────────────────────────────
#  Test block
# ───────────────────────────────────────────────────────
MPC = 3
base_conditions = [(m, e) for m in MODES for e in EXPECT]
n_total_trials = len(base_conditions) * MPC

# No catch trials for now - focus on expectation effects
n_main_per_cond = MPC

# Create lists of trial types for each condition
all_trials = []
for cond in base_conditions:
    all_trials.extend([(cond, "") for i in range(n_main_per_cond)])

# Create and assign counterbalanced targets
n_total_final = len(all_trials)
n_half = n_total_final // 2
targets = ['square'] * n_half + ['dot'] * (n_total_final - n_half)
rng.shuffle(targets)

# Combine conditions, types, and targets, then shuffle
combined = [(cond, ctype, target) for (cond, ctype), target in zip(all_trials, targets)]
rng.shuffle(combined)

msg.text = "MAIN BLOCK – Press any key."; msg.draw(); win.flip(); wait_keys()
for cond, ctype, a_target in combined:
    trial_counter += 1
    angle_bias, expect_level = cond
    res = run_trial(trial_counter, "test", angle_bias, expect_level, "medium", 
                   catch_type=ctype, target_shape=a_target)

    for k, v in res.items(): thisExp.addData(k, v)
    thisExp.nextEntry()
    if trial_counter % BREAK_EVERY == 0 and not SIMULATE and trial_counter < len(combined):
        msg.text = f"Break – press any key to continue."; msg.draw(); win.flip(); wait_keys()

# ───────────────────────────────────────────────────────
#  End of Experiment
# ───────────────────────────────────────────────────────
msg.text = "Thank you – task complete! Press any key."; msg.draw(); win.flip(); wait_keys()
win.close(); core.quit() 