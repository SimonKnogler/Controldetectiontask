#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
control_detection_task_v16.2_kinematics.py
─────────────────────────────────────────────
v16.2 Changes:
• Now records frame-by-frame kinematics (mouse and shape positions) for every trial.
• Saves kinematics to a separate CSV file for detailed analysis.
• Catch trials are now fully counterbalanced across the 4 main conditions.
• File naming is simplified to CDT_[participant_id].csv.
• Stimuli are now presented side-by-side at trial start, awaiting mouse movement.
• All previous fixes (single window, single save, clickable slider) are included.
"""

import os, sys, math, random, pathlib, datetime, atexit, hashlib, json
import numpy as np
import pandas as pd # Added for saving kinematics data
from psychopy import visual, event, core, data, gui


SIMULATE = True

# ───────────────────────────────────────────────────────
#  Global variable for kinematics data
# ───────────────────────────────────────────────────────
kinematics_data = []
kinematics_csv_path = "" # Will be defined later


# ───────────────────────────────────────────────────────
#  Auto‐save on quit (now saves both files)
# ───────────────────────────────────────────────────────
_saved = False
def _save():
    global _saved
    if not _saved:
        # Save main trial-by-trial data
        thisExp.saveAsWideText(csv_path)
        print("Main data auto‐saved ➜", csv_path)

        # Save frame-by-frame kinematics data
        if kinematics_data:
            kinematics_df = pd.DataFrame(kinematics_data)
            kinematics_df.to_csv(kinematics_csv_path, index=False)
            print("Kinematics data auto‐saved ➜", kinematics_csv_path)

        _saved = True
atexit.register(_save)

# ───────────────────────────────────────────────────────
#  Participant dialog
# ───────────────────────────────────────────────────────
expName = "ControlDetection_v16.2_kinematics" # Updated version name
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
#  Style‐matching setup: load clusters & scaler params
# ───────────────────────────────────────────────────────
LIB_NAME = "/Users/simonknogler/Desktop/PhD/Buildermolib/core_pool.npy"
motion_pool  = np.load(LIB_NAME)
SNIP_LEN     = motion_pool.shape[1]
TOTAL_SNIPS  = motion_pool.shape[0]
print(f"Loaded {TOTAL_SNIPS} snippets × {SNIP_LEN} frames from {LIB_NAME}")
K_CLUST = 6
SNIPS_PER_CLUSTER = TOTAL_SNIPS // K_CLUST
CLUSTER_POOLS = [
    motion_pool[i * SNIPS_PER_CLUSTER : (i + 1) * SNIPS_PER_CLUSTER]
    for i in range(K_CLUST)
]
with open("scaler_params.json", "r") as f:
    scp = json.load(f)
scaler_mean = np.array(scp["mean"], dtype=np.float32)
scaler_std  = np.array(scp["std"],  dtype=np.float32)
with open("/Users/simonknogler/Desktop/PhD/Controldetectiontask/Motion Library/cluster_centroids.json", "r") as f:
    CLUSTER_CENTROIDS = np.array(json.load(f), dtype=np.float32)
cluster_id = None
seed = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(),16) & 0xFFFFFFFF
rng  = np.random.default_rng(seed)

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
#  Helper: assign a participant to the closest style cluster
# ───────────────────────────────────────────────────────
def assign_cluster(demo_mean_speed, demo_mean_turn, demo_straightness):
    demo_vector = np.array([
        demo_mean_speed, 0.0, demo_mean_turn, 0.0, demo_straightness, 0.0
    ], dtype=np.float32)
    demo_scaled = (demo_vector - scaler_mean) / (scaler_std + 1e-9)
    dists = np.linalg.norm(CLUSTER_CENTROIDS - demo_scaled.reshape(1, -1), axis=1)
    return int(np.argmin(dists))

# ───────────────────────────────────────────────────────
#  Paths & ExperimentHandler
# ───────────────────────────────────────────────────────
root = pathlib.Path.cwd() / "data"; root.mkdir(exist_ok=True)

# New, simplified file naming convention
participant_id = expInfo['participant']
base_filename = f"CDT_{participant_id}"
csv_path = root / f"{base_filename}.csv"
kinematics_csv_path = root / f"{base_filename}_kinematics.csv"

# Prevent overwriting data by adding a number if the file already exists
i = 1
while csv_path.exists():
    new_filename = f"CDT_{participant_id}_{i}"
    csv_path = root / f"{new_filename}.csv"
    kinematics_csv_path = root / f"{new_filename}_kinematics.csv"
    i += 1

thisExp = data.ExperimentHandler(
    name=expName, extraInfo=expInfo,
    savePickle=False, saveWideText=False,
    dataFileName=str(root / base_filename) # Use base name for internal reference
)

# ───────────────────────────────────────────────────────
#  Window & stimuli
# ───────────────────────────────────────────────────────
win         = visual.Window((1920,1080), fullscr=not SIMULATE, color=[0.5]*3, units="pix")
square      = visual.Rect(win, 40, 40, fillColor="black", lineColor="black")
dot         = visual.Circle(win, 20, fillColor="black", lineColor="black")
fix         = visual.TextStim(win, "+", color="white", height=30)
msg         = visual.TextStim(win, "", color="white", height=26, wrapWidth=1000)
feedbackTxt = visual.TextStim(win, "", color="black", height=80)

confine = lambda p, l=250: p if (r := math.hypot(*p)) <= l else (p[0]*l/r, p[1]*l/r)
rotate  = lambda vx, vy, a: (
    vx * math.cos(math.radians(a)) - vy * math.sin(math.radians(a)),
    vx * math.sin(math.radians(a)) + vy * math.cos(math.radians(a))
)

snippet_queue = []

# ───────────────────────────────────────────────────────
#  Demo Trial
# ───────────────────────────────────────────────────────
def demo():
    global avg_demo_speed, cluster_id
    prop_demo = 0.80
    msg.text = "30‐s DEMO – one shape mostly follows your mouse.\nPress any key."
    msg.draw(); win.flip(); wait_keys()
    fix.draw(); win.flip(); core.wait(0.5)
    sqX, sqY = np.random.uniform(-200, 200, 2)
    square.pos, dot.pos = (sqX, sqY), (-sqX, -sqY)
    (ou_x, ou_y), _ = sample_snippet()
    demo_positions = []; speeds = []
    mouse = SimulatedMouse() if SIMULATE else event.Mouse(win=win, visible=True)
    mouse.setPos((0, 0))
    last = (0, 0); frame = 0; clk = core.Clock()
    while clk.getTime() <5.0:
        x, y = mouse.getPos()
        demo_positions.append((x, y))
        dx, dy = x - last[0], y - last[1]
        last = (x, y); speeds.append(math.hypot(dx, dy))
        ou_dx, ou_dy = ou_x[frame % SNIP_LEN], ou_y[frame % SNIP_LEN]
        frame += 1
        mag_m = math.hypot(dx, dy); mag_o = math.hypot(ou_dx, ou_dy)
        if mag_o > 0: ou_dx, ou_dy = ou_dx / mag_o * mag_m, ou_dy / mag_o * mag_m
        tdx = prop_demo * dx + (1 - prop_demo) * ou_dx
        tdy = prop_demo * dy + (1 - prop_demo) * ou_dy
        global vt, vd
        if 'vt' in globals():
            vt = 0.8 * vt + 0.2 * np.array([tdx, tdy])
            vd = 0.8 * vd + 0.2 * np.array([ou_dx, ou_dy])
        else:
            vt = np.array([tdx, tdy]); vd = np.array([ou_dx, ou_dy])
        square.pos = confine(tuple(square.pos + vt))
        dot.pos    = confine(tuple(dot.pos + vd))
        square.draw(); dot.draw(); win.flip()
    avg_demo_speed = float(np.mean(speeds))
    demo_arr = np.array(demo_positions, dtype=np.float32)
    deltas = np.diff(demo_arr, axis=0); demo_speeds = np.linalg.norm(deltas, axis=1)
    demo_mean_speed = float(np.mean(demo_speeds))
    unit = deltas / (demo_speeds.reshape(-1, 1) + 1e-9)
    dotp = np.einsum('ij,ij->i', unit[:-1], unit[1:])
    dotp = np.clip(dotp, -1.0, 1.0); angles = np.arccos(dotp)
    demo_mean_turn = float(np.mean(angles))
    net_disp = demo_arr[-1] - demo_arr[0]; net_mag  = np.linalg.norm(net_disp)
    path_len = np.sum(demo_speeds)
    demo_straightness = float(net_mag / (path_len + 1e-9))
    cluster_id = assign_cluster(demo_mean_speed, demo_mean_turn, demo_straightness)
    print(f"Participant assigned to cluster {cluster_id}")

def sample_snippet():
    global snippet_queue
    if not snippet_queue: snippet_queue = rng.permutation(len(motion_pool)).tolist()
    idx = snippet_queue.pop()
    return motion_pool[idx].astype(np.float32).T, idx

def sample_snippet_scaled():
    local_idx = rng.integers(0, SNIPS_PER_CLUSTER)
    snip_abs = CLUSTER_POOLS[cluster_id][local_idx].astype(np.float32)
    vel = np.diff(snip_abs, axis=0)
    vel = np.vstack((vel[0, :].reshape(1, 2), vel))
    speeds = np.linalg.norm(vel, axis=1)
    snippet_mean_speed = float(np.mean(speeds))
    scale_factor = avg_demo_speed / (snippet_mean_speed + 1e-9)
    vel_scaled = vel * scale_factor
    return vel_scaled.T, local_idx

demo()

# ───────────────────────────────────────────────────────
#  Staircase and Trial Parameters
# ───────────────────────────────────────────────────────
MODES=[0,90]; EXPECT=["low","high"]
TARGET={"low":0.60,"high":0.80}; BASE=0.05
STEP_DOWN={e:BASE for e in EXPECT}
STEP_UP   ={e:(TARGET[e]/(1-TARGET[e]))*BASE for e in EXPECT}
PROP={m:{"low":0.50,"high":0.80} for m in MODES}
MED_PROP  = {m:(PROP[m]["low"]+PROP[m]["high"])/2 for m in MODES}
TEST_PROP = MED_PROP.copy()
BREAK_EVERY=15; LOWPASS=0.8; OFFSET_X = 150 # <-- Added offset constant
CONF_KEYS=["1","2","3","4"]
SHAPE_FROM_KEY={"1":"square","2":"square","3":"dot","4":"dot"}
CONF_FROM_KEY={k:i+1 for i,k in enumerate(CONF_KEYS)}

# ───────────────────────────────────────────────────────
#  Trial function (now logs kinematics)
# ───────────────────────────────────────────────────────
def run_trial(trial_num, phase, angle_bias, expect_level, mode, catch_type="", target_shape=None):
    if catch_type == "full": prop = 1.0
    elif mode == "true": prop = PROP[angle_bias][expect_level]
    else: prop = TEST_PROP[angle_bias]

    cue = low_col if expect_level == "low" else high_col
    fix.color = cue; square.fillColor = square.lineColor = cue; dot.fillColor = dot.lineColor = cue

    fix.draw(); win.flip(); core.wait(0.5)

    # --- NEW: Initial stimulus presentation ---
    # Randomly determine which shape starts on the left
    left_shape = random.choice(['square', 'dot'])
    if left_shape == 'square':
        square.pos = (-OFFSET_X, 0)
        dot.pos = (OFFSET_X, 0)
    else:
        square.pos = (OFFSET_X, 0)
        dot.pos = (-OFFSET_X, 0)
    
    # Draw the static shapes and wait for movement to start the trial
    square.draw(); dot.draw(); win.flip()
    
    mouse = SimulatedMouse() if SIMULATE else event.Mouse(win=win, visible=True)
    mouse.setPos((0, 0))
    last = mouse.getPos()
    while True:
        # Keep drawing the static shapes while waiting
        square.draw(); dot.draw(); win.flip()
        x, y = mouse.getPos()
        if math.hypot(x - last[0], y - last[1]) > 0 or SIMULATE: break
        if not SIMULATE and event.getKeys(["escape"]): _save(); core.quit()
    # --- END of new presentation logic ---

    target = target_shape if target_shape is not None else random.choice(["square", "dot"])
    (ou_x, ou_y), snip_id = sample_snippet_scaled()

    trial_kinematics = []
    clk = core.Clock(); frame = 0
    vt = vd = np.zeros(2, np.float32)

    while clk.getTime() < 3.0:
        x, y = mouse.getPos()
        dx, dy = x - last[0], y - last[1]
        last = (x, y)
        dx, dy = rotate(dx, dy, angle_bias)
        ou_dx, ou_dy = ou_x[frame % SNIP_LEN], ou_y[frame % SNIP_LEN]
        frame += 1
        mag_m = math.hypot(dx, dy); mag_o = math.hypot(ou_dx, ou_dy)
        if mag_o > 0: ou_dx, ou_dy = ou_dx / mag_o * mag_m, ou_dy / mag_o * mag_m
        tdx = prop * dx + (1 - prop) * ou_dx
        tdy = prop * dy + (1 - prop) * ou_dy
        vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
        vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ou_dx, ou_dy])

        if target == "square":
            square.pos = confine(tuple(square.pos + vt))
            dot.pos    = confine(tuple(dot.pos + vd))
        else:
            dot.pos    = confine(tuple(dot.pos + vt))
            square.pos = confine(tuple(square.pos + vd))

        trial_kinematics.append({'timestamp': clk.getTime(), 'frame': frame, 'mouse_x': x, 'mouse_y': y,
            'square_x': square.pos[0], 'square_y': square.pos[1], 'dot_x': dot.pos[0], 'dot_y': dot.pos[1]
        })
        square.draw(); dot.draw(); win.flip()

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
        feedbackTxt.text = "✓" if correct else "✗"
        feedbackTxt.draw(); win.flip(); core.wait(1.0)
        win.flip(); core.wait(0.5)

    rating = np.nan
    if phase == "test":
        if SIMULATE: rating = float(rng.integers(0, 101))
        else:
            mouse = event.Mouse(win=win, visible=True)
            slider = visual.Slider(win, pos=(0, -250), size=(600, 40), ticks=(0, 100), labels=("0","100"), granularity=1, style='rating', labelHeight=24, color='white', fillColor='white')
            msg.text = "How much control did you feel?"
            msg.draw(); slider.draw(); win.flip()
            while slider.rating is None:
                slider.draw(); msg.draw(); win.flip()
                if not SIMULATE and event.getKeys(["escape"]): _save(); core.quit()
            rating = float(slider.rating); core.wait(0.2)

    for frame_data in trial_kinematics:
        frame_data.update({'participant': expInfo['participant'], 'session': expInfo['session'], 'trial_num': trial_num,
            'phase': phase, 'angle_bias': angle_bias, 'expect_level': expect_level, 'prop_used': prop,
            'conf_level': conf_lvl, 'agency_rating': rating})
        kinematics_data.append(frame_data)

    if phase == "practice" and catch_type == "":
        if correct: PROP[angle_bias][expect_level] = max(0, PROP[angle_bias][expect_level] - STEP_DOWN[expect_level])
        else: PROP[angle_bias][expect_level] = min(1, PROP[angle_bias][expect_level] + STEP_UP[expect_level])

    return dict(
        snippet_id=snip_id, catch_type=catch_type, phase=phase, angle_bias=angle_bias,
        expect_level=expect_level, true_shape=target, resp_shape=resp_shape,
        conf_level=conf_lvl, accuracy=correct, rt_choice=rt_choice,
        agency_rating=rating, prop_used=prop
    )

# ───────────────────────────────────────────────────────
#  Practice block
# ───────────────────────────────────────────────────────
PPC = 3
practice = [(m, e) for m in MODES for e in EXPECT] * PPC
random.shuffle(practice)
msg.text = "PRACTICE – Press any key."; msg.draw(); win.flip(); wait_keys()
trial_counter = 0
for m, e in practice:
    trial_counter += 1
    res = run_trial(trial_counter, "practice", m, e, "true")
    for k, v in res.items(): thisExp.addData(k, v)
    thisExp.nextEntry()

MED_PROP = {m: (PROP[m]["low"] + PROP[m]["high"]) / 2 for m in MODES}
TEST_PROP = MED_PROP.copy()

# ───────────────────────────────────────────────────────
#  Test block (with counterbalanced catch trials and targets)
# ───────────────────────────────────────────────────────
MPC = 20
base_conditions = [(m, e) for m in MODES for e in EXPECT]
n_total_trials = len(base_conditions) * MPC

# Calculate number of catch trials (~5%), ensuring it's divisible by 4
n_catch = int(n_total_trials * 0.05)
n_catch = (n_catch // 4) * 4 # Round down to nearest multiple of 4 for perfect balance
n_catch_per_cond = n_catch // 4
n_main_per_cond = MPC - n_catch_per_cond

# Create lists of trial types for each condition
all_trials = []
for cond in base_conditions:
    all_trials.extend([ (cond, "") for i in range(n_main_per_cond) ])
    all_trials.extend([ (cond, "full") for i in range(n_catch_per_cond) ])

# Create and assign counterbalanced targets
n_total_final = len(all_trials)
n_half = n_total_final // 2
targets = ['square'] * n_half + ['dot'] * (n_total_final - n_half)
rng.shuffle(targets)

# Combine conditions, types, and targets, then shuffle
combined = [ (cond, ctype, target) for (cond, ctype), target in zip(all_trials, targets) ]
rng.shuffle(combined)


msg.text = "MAIN BLOCK – Press any key."; msg.draw(); win.flip(); wait_keys()
for cond, ctype, a_target in combined:
    trial_counter += 1
    angle_bias, expect_level = cond
    res = run_trial(trial_counter, "test", angle_bias, expect_level, "medium", catch_type=ctype, target_shape=a_target)

    for k, v in res.items(): thisExp.addData(k, v)
    thisExp.nextEntry()
    if trial_counter % BREAK_EVERY == 0 and not SIMULATE and trial_counter < len(combined):
        msg.text = f"Break – press any key to continue."; msg.draw(); win.flip(); wait_keys()

# ───────────────────────────────────────────────────────
#  End of Experiment
# ───────────────────────────────────────────────────────
msg.text = "Thank you – task complete! Press any key."; msg.draw(); win.flip(); wait_keys()
win.close(); core.quit()
