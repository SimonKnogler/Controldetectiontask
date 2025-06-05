#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
control_detection_task_v16.1_style_matched.py
─────────────────────────────────────────────
Modified from v16.1 to implement style‐matching: 
• Loads motion snippets from CORE_POOL (6 clusters × 400 snippets each).
• After demo, computes participant’s movement features and assigns to a cluster.
• sample_snippet_scaled() draws only from that participant’s cluster and speed‐matches.
"""

import os, sys, math, random, pathlib, datetime, atexit, hashlib, json
import numpy as np
from psychopy import visual, event, core, data, gui 


SIMULATE = False

# ───────────────────────────────────────────────────────
#  Auto‐save on quit
# ───────────────────────────────────────────────────────
_saved = False
def _save():
    global _saved
    if not _saved:
        thisExp.saveAsWideText(csv_path)
        print("Data auto‐saved ➜", csv_path)
        _saved = True
atexit.register(_save)

# ───────────────────────────────────────────────────────
#  Participant dialog
# ───────────────────────────────────────────────────────
expName = "ControlDetection_v16.1_style_matched"
expInfo = {"participant":"", "session":"001"}
if not SIMULATE and not gui.DlgFromDict(expInfo, title=expName).OK:
    core.quit()
if SIMULATE:
    expInfo["participant"] = "SIM"

# counter‐balance cue colours
low_col, high_col = random.choice([("blue","green"),("green","blue")])
expInfo["low_precision_colour"]  = low_col
expInfo["high_precision_colour"] = high_col

# ───────────────────────────────────────────────────────
#  Style‐matching setup: load clusters & scaler params
# ───────────────────────────────────────────────────────

# Path to core_pool.npy (6 clusters × 400 snippets each, each snippet is 180×2)
LIB_NAME = "/Users/simonknogler/Desktop/PhD/Buildermolib/core_pool.npy"
motion_pool  = np.load(LIB_NAME)        # shape (2400, 180, 2)
SNIP_LEN     = motion_pool.shape[1]     # 180  (3 s @ 60 Hz)
TOTAL_SNIPS  = motion_pool.shape[0]     # 2400
print(f"Loaded {TOTAL_SNIPS} snippets × {SNIP_LEN} frames from {LIB_NAME}")

# Number of clusters (must match how core_pool.npy was built)
K_CLUST = 6
# Number of snippets per cluster (must match DESIRED_PER used in motionlib)
SNIPS_PER_CLUSTER = TOTAL_SNIPS // K_CLUST  # 2400 / 6 = 400

# Split motion_pool into 6 clusters: CLUSTER_POOLS[0] is cluster 0, etc.
CLUSTER_POOLS = [
    motion_pool[i * SNIPS_PER_CLUSTER : (i + 1) * SNIPS_PER_CLUSTER]
    for i in range(K_CLUST)
]
# Now CLUSTER_POOLS[c].shape == (400, 180, 2) for each c in 0..5

# Load scaler parameters (six means and six stds) saved by motionlib_create_filtered.py
with open("scaler_params.json", "r") as f:
    scp = json.load(f)
scaler_mean = np.array(scp["mean"], dtype=np.float32)   # shape (6,)
scaler_std  = np.array(scp["std"],  dtype=np.float32)   # shape (6,)

# Load cluster centroids (6 × 6 matrix) saved by motionlib_create_filtered.py
with open("cluster_centroids.json", "r") as f:
    CLUSTER_CENTROIDS = np.array(json.load(f), dtype=np.float32)  # shape (6, 6)

# Initialize cluster_id (will be set after demo)
cluster_id = None

# RNG seeded by participant ID for reproducibility
seed = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(),16) & 0xFFFFFFFF
rng  = np.random.default_rng(seed)

# ───────────────────────────────────────────────────────
#  Helper: assign a participant to the closest style cluster
# ───────────────────────────────────────────────────────
def assign_cluster(demo_mean_speed, demo_mean_turn, demo_straightness):
    """
    Given the participant's demo‐block features (mean_speed, mean_turn, straightness),
    build a 6‐D feature vector (with sd_speed, sd_turn, pause_frac = 0)
    z‐score it using scaler_mean & scaler_std, then find nearest centroid.
    Returns integer cluster index in [0 .. K_CLUST-1].
    """
    # Build raw 6‐D demo vector: [mean_speed, sd_speed=0, mean_turn, sd_turn=0, straightness, pause_frac=0]
    # We set sd_speed, sd_turn, pause_frac to 0 so that z-scored they correspond to mean.
    demo_vector = np.array([
        demo_mean_speed,   # raw mean_speed
        0.0,               # placeholder for sd_speed
        demo_mean_turn,    # raw mean_turn
        0.0,               # placeholder for sd_turn
        demo_straightness, # raw straightness
        0.0                # placeholder for pause_frac
    ], dtype=np.float32)

    # Z‐score using scaler parameters from motionlib pipeline
    demo_scaled = (demo_vector - scaler_mean) / (scaler_std + 1e-9)  # shape (6,)

    # Compute Euclidean distances to each cluster centroid (in z‐space)
    # CLUSTER_CENTROIDS[c] is 6‐D array for cluster c
    dists = np.linalg.norm(CLUSTER_CENTROIDS - demo_scaled.reshape(1, -1), axis=1)  # shape (6,)

    # Return index (0..5) of the nearest centroid
    return int(np.argmin(dists))


# ───────────────────────────────────────────────────────
#  One speed estimate per snippet (for optional debugging; not used below)
# ───────────────────────────────────────────────────────
# snippet_speeds = np.linalg.norm(motion_pool, axis=2).mean(1)  # array length 2400

# ───────────────────────────────────────────────────────
#  Paths & ExperimentHandler
# ───────────────────────────────────────────────────────
root = pathlib.Path.cwd() / "data"; root.mkdir(exist_ok=True)
file_stem = (
    f"{expName}_P{expInfo['participant']}_"
    f"S{expInfo['session']}_"
    f"{datetime.datetime.now():%Y%m%dT%H%M%S}"
)
csv_path = root / f"{file_stem}.csv"
thisExp = data.ExperimentHandler(
    name=expName, extraInfo=expInfo,
    savePickle=False, saveWideText=False,
    dataFileName=str(root / file_stem)
)

# ───────────────────────────────────────────────────────
#  Window & stimuli
# ───────────────────────────────────────────────────────
win         = visual.Window((1920,1080), fullscr=not SIMULATE,
                            color=[0.5]*3, units="pix")
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

# ───────────────────────────────────────────────────────
#  Draw without replacement: queue of global indices
# ───────────────────────────────────────────────────────
# We will draw from the entire pool if needed, 
# but in style‐matched sampling we only use CLUSTER_POOLS and compute velocities directly.
snippet_queue = []  # not used in style‐matched; kept for consistency


# ───────────────────────────────────────────────────────
#  30‐s demo: record positions, then compute features & assign cluster
# ───────────────────────────────────────────────────────
def demo():
    global avg_demo_speed, cluster_id
    # Proportion of actual mouse movement vs. snippet in demo (unchanged)
    prop_demo = 0.80

    # Instructions
    msg.text = "30‐s DEMO – one shape mostly follows your mouse.\nPress any key."
    msg.draw(); win.flip(); event.waitKeys()
    fix.draw(); win.flip(); core.wait(0.5)

    # Randomize initial positions of square & dot
    sqX, sqY = np.random.uniform(-200, 200, 2)
    square.pos, dot.pos = (sqX, sqY), (-sqX, -sqY)

    # For the demo, we use a random snippet as “background” (no scaling here)
    (ou_x, ou_y), _ = sample_snippet()

    # Prepare to record mouse positions (for feature extraction)
    demo_positions = []  # will store (x, y) each frame
    speeds = []          # store instantaneous speeds for demo

    # Initialize mouse & timing
    mouse = event.Mouse(win=win, visible=not SIMULATE); mouse.setPos((0, 0))
    last = (0, 0)
    frame = 0
    clk = core.Clock()
    # Run for 30 seconds (3 s × 10 blocks = 30 s)
    while clk.getTime() < 30.0:
        x, y = mouse.getPos()
        # Record current position
        demo_positions.append((x, y))

        # Compute mouse displacement
        dx, dy = x - last[0], y - last[1]
        last = (x, y)
        speeds.append(math.hypot(dx, dy))

        # Get snippet’s next velocity
        ou_dx, ou_dy = ou_x[frame % SNIP_LEN], ou_y[frame % SNIP_LEN]
        frame += 1

        # Scale snippet velocity to match mouse magnitude
        mag_m = math.hypot(dx, dy)
        mag_o = math.hypot(ou_dx, ou_dy)
        if mag_o > 0:
            ou_dx, ou_dy = ou_dx / mag_o * mag_m, ou_dy / mag_o * mag_m

        # Combine mouse and snippet
        tdx = prop_demo * dx + (1 - prop_demo) * ou_dx
        tdy = prop_demo * dy + (1 - prop_demo) * ou_dy

        # Exponential smoothing (unchanged)
        global vt, vd
        if 'vt' in globals():
            vt = 0.8 * vt + 0.2 * np.array([tdx, tdy])
            vd = 0.8 * vd + 0.2 * np.array([ou_dx, ou_dy])
        else:
            vt = np.array([tdx, tdy])
            vd = np.array([ou_dx, ou_dy])

        # Update positions and draw
        square.pos = confine(tuple(square.pos + vt))
        dot.pos    = confine(tuple(dot.pos + vd))
        square.draw(); dot.draw(); win.flip()

    # Compute average demo speed (px/frame)
    avg_demo_speed = float(np.mean(speeds))

    # Convert demo_positions to NumPy array for feature extraction
    demo_arr = np.array(demo_positions, dtype=np.float32)  # shape (T, 2)

    # Compute frame‐by‐frame displacements for demo
    deltas = np.diff(demo_arr, axis=0)                     # shape (T-1, 2)
    demo_speeds = np.linalg.norm(deltas, axis=1)           # length T-1
    demo_mean_speed = float(np.mean(demo_speeds))          # px/frame

    # Compute turning angles for demo
    unit = deltas / (demo_speeds.reshape(-1, 1) + 1e-9)     # shape (T-1, 2)
    dotp = np.einsum('ij,ij->i', unit[:-1], unit[1:])       # length T-2
    dotp = np.clip(dotp, -1.0, 1.0)
    angles = np.arccos(dotp)                                # length T-2
    demo_mean_turn = float(np.mean(angles))                 # in radians

    # Compute straightness for demo
    net_disp = demo_arr[-1] - demo_arr[0]                   # vector from start to end
    net_mag  = np.linalg.norm(net_disp)                     # scalar
    path_len = np.sum(demo_speeds)                          # total path length
    demo_straightness = float(net_mag / (path_len + 1e-9))

    # Now assign this participant to a style cluster
    cluster_id = assign_cluster(demo_mean_speed, demo_mean_turn, demo_straightness)
    print(f"Participant assigned to cluster {cluster_id}")

# Helper: sample_snippet (no scaling, for demo)
def sample_snippet():
    """
    Return (ou_x, ou_y), idx by drawing randomly from full motion_pool without replacement.
    Used only in the demo to provide background motion.
    """
    # If queue is empty, refill it
    global snippet_queue
    if not snippet_queue:
        snippet_queue = rng.permutation(len(motion_pool)).tolist()
    idx = snippet_queue.pop()
    # Return (2 × SNIP_LEN array), idx
    return motion_pool[idx].astype(np.float32).T, idx

# ───────────────────────────────────────────────────────
#  Modified sample_snippet_scaled() for style‐matching
# ───────────────────────────────────────────────────────
def sample_snippet_scaled():
    """
    Returns (velocities, snippet_index) where:
      - velocities: a (2, SNIP_LEN) float32 array of Δx, Δy per frame for one snippet
      - snippet_index: an integer 0..(SNIPS_PER_CLUSTER-1) indicating which snippet in this cluster
    Draws uniformly from CLUSTER_POOLS[cluster_id], computes per-frame deltas, scales
    to match avg_demo_speed, and returns the scaled velocities.
    """
    # 1· Pick a random local index within the assigned cluster
    local_idx = rng.integers(0, SNIPS_PER_CLUSTER)  # 0 ≤ local_idx < 400

    # 2· Extract that snippet's absolute positions (shape = (180, 2))
    snip_abs = CLUSTER_POOLS[cluster_id][local_idx].astype(np.float32)

    # 3· Compute per-frame displacements (Δx, Δy)
    vel = np.diff(snip_abs, axis=0)                   # shape = (179, 2)
    # Prepend first row so length = 180
    vel = np.vstack((vel[0, :].reshape(1, 2), vel))   # shape = (180, 2)

    # 4· Compute this snippet's raw mean speed (px/frame)
    speeds = np.linalg.norm(vel, axis=1)              # length = 180
    snippet_mean_speed = float(np.mean(speeds))       # scalar px/frame

    # 5· Compute scale factor to match participant's avg demo speed
    scale_factor = avg_demo_speed / (snippet_mean_speed + 1e-9)

    # 6· Scale velocities
    vel_scaled = vel * scale_factor                   # shape = (180, 2)

    # 7· Transpose to shape (2, 180) so downstream code can do: dx, dy = velocities[:, frame]
    vel_scaled = vel_scaled.T                         # now shape = (2, 180)

    # 8· Return velocities and the local index (0..399)
    return vel_scaled, local_idx

# ───────────────────────────────────────────────────────
#  Window & stimuli remain the same
# ───────────────────────────────────────────────────────
win         = visual.Window((1920,1080), fullscr=not SIMULATE,
                            color=[0.5]*3, units="pix")
square      = visual.Rect(win, 40, 40, fillColor="black", lineColor="black")
dot         = visual.Circle(win, 20, fillColor="black", lineColor="black")
fix         = visual.TextStim(win, "+", color="white", height=30)
msg         = visual.TextStim(win, "", color="white", height=26, wrapWidth=1000)
feedbackTxt = visual.TextStim(win, "", color="black", height=80)

# Re‐define these lambdas since we redrew window objects
confine = lambda p, l=250: p if (r := math.hypot(*p)) <= l else (p[0]*l/r, p[1]*l/r)
rotate  = lambda vx, vy, a: (
    vx * math.cos(math.radians(a)) - vy * math.sin(math.radians(a)),
    vx * math.sin(math.radians(a)) + vy * math.cos(math.radians(a))
)

# ───────────────────────────────────────────────────────
#  30‐s demo: get participant cluster assignment
# ───────────────────────────────────────────────────────
demo()   # After this call, avg_demo_speed and cluster_id are set

# ───────────────────────────────────────────────────────
#  Staircase parameters (unchanged)
# ───────────────────────────────────────────────────────
MODES=[0,90]; EXPECT=["low","high"]
TARGET={"low":0.60,"high":0.80}; BASE=0.05
STEP_DOWN={e:BASE for e in EXPECT}
STEP_UP   ={e:(TARGET[e]/(1-TARGET[e]))*BASE for e in EXPECT}
PROP={m:{"low":0.50,"high":0.80} for m in MODES}
MED_PROP  = {m:(PROP[m]["low"]+PROP[m]["high"])/2 for m in MODES}
TEST_PROP = MED_PROP.copy()

BREAK_EVERY=3; LOWPASS=0.8
CONF_KEYS=["1","2","3","4"]
SHAPE_FROM_KEY={"1":"square","2":"square","3":"dot","4":"dot"}
CONF_FROM_KEY={k:i+1 for i,k in enumerate(CONF_KEYS)}

# ───────────────────────────────────────────────────────
#  Trial function (uses dynamic SNIP_LEN, style‐matched sampling)
# ───────────────────────────────────────────────────────
def run_trial(phase, angle_bias, expect_level, mode, catch_type=""):
    # determine proportion of mouse vs. snippet
    if catch_type == "full":
        prop = 1.0
    elif mode == "true":
        prop = PROP[angle_bias][expect_level]
    else:
        prop = TEST_PROP[angle_bias]

    cue = low_col if expect_level == "low" else high_col
    fix.color = cue
    square.fillColor = square.lineColor = cue
    dot.fillColor    = dot.lineColor    = cue

    fix.draw(); win.flip(); core.wait(0.5)
    square.draw(); dot.draw(); win.flip()

    # Randomize start positions
    sqX, sqY = np.random.uniform(-200, 200, 2)
    square.pos, dot.pos = (sqX, sqY), (-sqX, -sqY)
    target = random.choice(["square", "dot"])

    # SAMPLE STYLE‐MATCHED SNIPPET (velocities, local_idx)
    (ou_x, ou_y), snip_id = sample_snippet_scaled()

    # Initialize mouse
    mouse = event.Mouse(win=win, visible=not SIMULATE); mouse.setPos((0, 0))
    last = mouse.getPos()
    # Wait for movement to start (skip 0‐movement frames)
    while True:
        x, y = mouse.getPos()
        if math.hypot(x - last[0], y - last[1]) > 0:
            break
        if event.getKeys(["escape"]):
            _save(); core.quit()

    clk = core.Clock(); frame = 0
    vt = vd = np.zeros(2, np.float32)
    # Main trial loop (3 seconds)
    while clk.getTime() < 3.0:
        x, y = mouse.getPos()
        dx, dy = x - last[0], y - last[1]
        last = (x, y)
        # Apply angular bias
        dx, dy = rotate(dx, dy, angle_bias)

        # Get snippet velocity for this frame
        ou_dx, ou_dy = ou_x[frame % SNIP_LEN], ou_y[frame % SNIP_LEN]
        frame += 1

        # Scale snippet velocity to match mouse magnitude
        mag_m = math.hypot(dx, dy)
        mag_o = math.hypot(ou_dx, ou_dy)
        if mag_o > 0:
            ou_dx, ou_dy = ou_dx / mag_o * mag_m, ou_dy / mag_o * mag_m

        # Combine signals
        tdx = prop * dx + (1 - prop) * ou_dx
        tdy = prop * dy + (1 - prop) * ou_dy

        # Exponential smoothing
        vt = LOWPASS * vt + (1 - LOWPASS) * np.array([tdx, tdy])
        vd = LOWPASS * vd + (1 - LOWPASS) * np.array([ou_dx, ou_dy])

        # Update positions of square & dot
        if target == "square":
            square.pos = confine(tuple(square.pos + vt))
            dot.pos    = confine(tuple(dot.pos + vd))
        else:
            dot.pos    = confine(tuple(dot.pos + vt))
            square.pos = confine(tuple(square.pos + vd))

        square.draw(); dot.draw(); win.flip()

    # Decision & confidence rating (unchanged)
    msg.text = (
        "Which shape did you control & how confident?\n"
        "1 Sq(guess) 2 Sq(conf) 3 Dot(conf) 4 Dot(guess)"
    )
    msg.draw(); win.flip()
    t0 = core.getTime()
    key = event.waitKeys(keyList=CONF_KEYS + ["escape"])[0]
    rt_choice = core.getTime() - t0
    if key == "escape":
        _save(); core.quit()
    resp_shape = SHAPE_FROM_KEY[key]
    conf_lvl = CONF_FROM_KEY[key]
    correct = int(resp_shape == target)

    if phase == "practice":
        feedbackTxt.text = "✓" if correct else "✗"
        feedbackTxt.draw(); win.flip(); core.wait(1.0)
        win.flip(); core.wait(0.5)

    rating = np.nan
    if phase == "test":
        slider = visual.Slider(
            win, pos=(0, -250), size=(600, 40),
            ticks=(0, 100), labels=("0","100"),
            granularity=1, style='rating',
            labelHeight=24, color='white', fillColor='white'
        )
        msg.text = "How much control did you feel?"
        msg.draw(); slider.draw(); win.flip()
        while slider.rating is None:
            slider.draw(); msg.draw(); win.flip()
            if event.getKeys(["escape"]):
                _save(); core.quit()
        rating = float(slider.rating); core.wait(0.2)

    if phase == "practice" and catch_type == "":
        if correct:
            PROP[angle_bias][expect_level] = max(
                0, PROP[angle_bias][expect_level] - STEP_DOWN[expect_level]
            )
        else:
            PROP[angle_bias][expect_level] = min(
                1, PROP[angle_bias][expect_level] + STEP_UP[expect_level]
            )

    return dict(
        snippet_id = snip_id, catch_type=catch_type, phase=phase,
        angle_bias=angle_bias, expect_level=expect_level,
        true_shape=target, resp_shape=resp_shape,
        conf_level=conf_lvl, accuracy=correct,
        rt_choice=rt_choice, agency_rating=rating,
        prop_used=prop
    )

# ───────────────────────────────────────────────────────
#  Practice block (unchanged logic)
# ───────────────────────────────────────────────────────
PPC = 3
practice = [(m, e) for m in MODES for e in EXPECT] * PPC
random.shuffle(practice)
msg.text = "PRACTICE – Press any key."; msg.draw(); win.flip(); event.waitKeys()
for m, e in practice:
    res = run_trial("practice", m, e, "true")
    for k, v in res.items():
        thisExp.addData(k, v)
    thisExp.nextEntry()

# Update TEST_PROP (unchanged)
MED_PROP = {m: (PROP[m]["low"] + PROP[m]["high"]) / 2 for m in MODES}
TEST_PROP = MED_PROP.copy()

# ───────────────────────────────────────────────────────
#  Test block (unchanged logic)
# ───────────────────────────────────────────────────────
MPC = 2
main_trials = [(m, e) for m in MODES for e in EXPECT] * MPC
n_catch = int(len(main_trials) * 0.05)
catch_trials = ["full"] * n_catch

conds = main_trials + catch_trials
types = [""] * len(main_trials) + catch_trials
combined = list(zip(conds, types)); rng.shuffle(combined)

msg.text = "MAIN BLOCK – Press any key."; msg.draw(); win.flip(); event.waitKeys()
t = 0
for cond, ctype in combined:
    if ctype == "full":
        res = run_trial("test", cond[0], cond[1], "medium", catch_type="full")
    else:
        res = run_trial("test", cond[0], cond[1], "medium", catch_type="")
    for k, v in res.items():
        thisExp.addData(k, v)
    thisExp.nextEntry()
    t += 1
    if t % BREAK_EVERY == 0 and not SIMULATE:
        msg.text = "Break – press any key"; msg.draw(); win.flip(); event.waitKeys()

# ───────────────────────────────────────────────────────
#  Save & quit
# ───────────────────────────────────────────────────────
thisExp.saveAsWideText(csv_path)
print("Saved ➜", csv_path)
msg.text = "Thank you – task complete! Press any key."; msg.draw(); win.flip(); event.waitKeys()
win.close(); core.quit()
