#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Control-Detection Task — weighted up–down v2.3  (2025-05-05)
----------------------------------------------------------
Implements Kaernbach’s (1991) weighted up–down to target
60% on “low” (hard) and 80% on “high” (easy) trials.
"""

# ====================================================
# 0  Imports & SIMULATE toggle
# ====================================================
import os                             # filesystem operations
import sys                            # system functions, for exit handling
import math                           # math utilities: hypot, sin, cos, radians
import random                         # random choices and shuffling
import pathlib                        # easier path handling
import datetime                       # timestamping filenames
import atexit                         # register exit handlers
import numpy as np                   # numeric arrays and random
from psychopy import visual, event, core, data, gui  # PsychoPy components

# Toggle this to True for autorun (no real window), False for actual experiment
SIMULATE = False                     # True = simulate; False = real run

# ====================================================
# 0.1  Exit handler to save data on any quit
# ====================================================
_saved = False                       # flag to avoid double-saving

def _save_data_on_exit():
    """Automatically save data if core.quit() or sys.exit() is called."""
    global _saved
    if not _saved:
        try:
            thisExp.saveAsWideText(csv_path)           # write .csv
            print(f"Data auto-saved to {csv_path}")    # log console message
            _saved = True                             # prevent re-saving
        except Exception as e:
            print(f"Auto-save failed: {e}")           # report errors

atexit.register(_save_data_on_exit)     # ensure handler runs on exit

# ====================================================
# 1  Experiment Info & counter-balancing
# ====================================================
expName = "ControlDetection_WP1_v2"      # experiment identifier
expInfo = {"participant":"", "session":"001"}  # dialog fields

if not SIMULATE:
    # show dialog to enter participant ID/session; quit if cancelled
    dlg = gui.DlgFromDict(expInfo, title=expName)
    if not dlg.OK:
        core.quit()                       # calls exit handler
else:
    expInfo["participant"] = "SIM"       # automatic label in simulate mode

# randomly assign which colour is “low” vs “high” precision cue
if random.random() < 0.5:
    low_col, high_col = "blue", "green"
else:
    low_col, high_col = "green", "blue"
expInfo["low_precision_colour"]  = low_col
expInfo["high_precision_colour"] = high_col

# ====================================================
# 2  Paths & ExperimentHandler
# ====================================================
# create a data folder in the current working directory
root = pathlib.Path.cwd() / "data"
root.mkdir(exist_ok=True)

# filename stem includes expName, participant, session, timestamp
file_stem = (
    f"{expName}_P{expInfo['participant']}_"
    f"S{expInfo['session']}_"
    f"{datetime.datetime.now():%Y%m%dT%H%M%S}"
)
csv_path = root / f"{file_stem}.csv"     # full path to CSV file

# initialize PsychoPy ExperimentHandler (we’ll save wide text manually)
thisExp = data.ExperimentHandler(
    name=expName,
    extraInfo=expInfo,
    savePickle=False,
    saveWideText=False,
    dataFileName=str(root / file_stem)
)

# ====================================================
# 3  Window & clock
# ====================================================
win = visual.Window(
    size=(1920,1080),                    # resolution in pixels
    fullscr=not SIMULATE,                # fullscreen if not simulating
    color=[0.5]*3,                       # mid-gray background
    units="pix"                          # use pixel units
)
globalClock = core.Clock()               # track overall time

# ====================================================
# 4  Stimuli definitions
# ====================================================
square = visual.Rect(win,
                     width=40, height=40,
                     fillColor="black",
                     lineColor="black")     # square stimulus

dot = visual.Circle(win,
                    radius=20,
                    fillColor="black",
                    lineColor="black")       # dot stimulus

fix = visual.TextStim(win,
                      text="+",
                      color="white",
                      height=30)               # fixation cross

msg = visual.TextStim(win,
                      text="",
                      color="white",
                      height=26,
                      wrapWidth=1000)          # instructions & ratings

# ====================================================
# 5  Helper functions
# ====================================================
# confine position p=(x,y) to circle radius l (default=250 px)
confine = lambda p, l=250: (
    p if (r := math.hypot(*p)) <= l else (p[0]*l/r, p[1]*l/r)
)
# rotate vector (vx,vy) by angle a (degrees)
rotate = lambda vx, vy, a: (
    vx*math.cos(math.radians(a)) - vy*math.sin(math.radians(a)),
    vx*math.sin(math.radians(a)) + vy*math.cos(math.radians(a))
)

def OU_snippet(n=300, theta=0.2, sigma=0.8, dt=1.0,
               lin_bias=(0,0), ang_bias=0.0):
    """
    Generate an Ornstein–Uhlenbeck snippet of length n for background motion.
    theta: mean-reversion strength
    sigma: noise scale
    lin_bias: constant linear drift
    ang_bias: constant angular drift
    """
    lin_bias = np.asarray(lin_bias, float)  # convert to array
    v = np.zeros((n,2))                     # velocity time series
    for t in range(1, n):
        # OU update: previous - theta*v*dt + noise + drift
        v[t] = (v[t-1]
                - theta * v[t-1] * dt
                + sigma * np.random.randn(2) * math.sqrt(dt)
                + lin_bias * dt)
        # convert to polar, add angular bias, convert back
        spd = math.hypot(*v[t])
        ang = math.atan2(v[t,1], v[t,0]) + ang_bias * dt
        v[t] = spd * np.array([math.cos(ang), math.sin(ang)])
    return v

# ====================================================
# 6  Pre-experiment demo (bias estimation)
# ====================================================
mouse = event.Mouse(win=win, visible=not SIMULATE)  # real or dummy mouse
mouse.setPos((0,0))                                 # start at center
msg.text = ("30-s DEMO – move the mouse; one shape will mostly follow you.\n"
            "Press any key.")
msg.draw(); win.flip(); event.waitKeys()            # show instructions
fix.draw(); win.flip(); core.wait(0.5)              # show fixation

# random start positions for square and dot (opposite sides)
sqX, sqY = np.random.uniform(-200,200,2)
dotX, dotY = -sqX, -sqY
square.pos, dot.pos = (sqX, sqY), (dotX, dotY)
control_shape = random.choice(["square","dot"])     # which shape follows mouse

traj = []               # to record mouse deltas
clk = core.Clock()      # clock for 30-s duration
last = mouse.getPos()   # last mouse position
frame = 0
snip = OU_snippet(1800) # OU snippet for background motion

while clk.getTime() < 5:  # 30 seconds
    x, y = mouse.getPos()             # current mouse pos
    dx, dy = x-last[0], y-last[1]     # mouse delta
    last = (x, y)
    ou_dx, ou_dy = snip[frame % 1800] # get OU motion
    frame += 1
    traj.append([dx, dy])             # record delta
    # move controlled shape by mouse, other by OU
    if control_shape == "square":
        square.pos = confine((square.pos[0]+dx,
                               square.pos[1]+dy))
        dot.pos    = confine((dot.pos[0]+ou_dx,
                               dot.pos[1]+ou_dy))
    else:
        dot.pos    = confine((dot.pos[0]+dx,
                               dot.pos[1]+dy))
        square.pos = confine((square.pos[0]+ou_dx,
                               square.pos[1]+ou_dy))
    square.draw(); dot.draw(); win.flip()

traj = np.array(traj)  # convert to array
lin_bias = traj.mean(0)  # estimate linear bias
ang_bias = (np.nanmean(np.diff(
            np.arctan2(traj[:,1], traj[:,0])))
            if traj.size else 0.0)

# ====================================================
# 7  Build OU library for trials
# ====================================================
LIB_N, LIB_LEN = 1000, 300
ou_lib = np.array([
    OU_snippet(LIB_LEN, lin_bias=lin_bias, ang_bias=ang_bias)
    for _ in range(LIB_N)
])

# ====================================================
# 8  Staircase parameters: weighted up–down (Kaernbach 1991)
# ====================================================
MODES = [0, 90]                        # rotation biases
EXPECT = ["low", "high"]               # cue levels

TARGET = {"low":  0.60,                # target 60% correct on “low” (hard)
          "high": 0.80}                # target 80% correct on “high” (easy)

BASE_STEP = 0.05                       # base downward step size

# step sizes: down = Δ, up = (p/(1-p)) * Δ
STEP_DOWN = {e: BASE_STEP for e in EXPECT}
STEP_UP   = {
    e: (TARGET[e] / (1 - TARGET[e])) * BASE_STEP
    for e in EXPECT
}

# initial self-motion proportions
PROP = {
    m: {"low": 0.50, "high": 0.80}
    for m in MODES
}

# placeholder for “medium” trials (computed after practice)
MEDIUM_PROP_MODE = {
    m: (PROP[m]["low"] + PROP[m]["high"]) / 2
    for m in MODES
}

BREAK_EVERY = 3                       # break every 3 trials
LOWPASS = 0.8                         # low-pass filter constant
CONF_KEYS = ["1","2","3","4"]         # response keys
SHAPE_FROM_KEY = {                    # map number key to shape
    "1": "square",                    # “1” → square guess
    "2": "square",                    # “2” → square confident
    "3": "dot",                       # “3” → dot confident
    "4": "dot"                        # “4” → dot guess
}
CONF_FROM_KEY = {k: i+1 for i,k in enumerate(CONF_KEYS)}  # map keys to confidence levels

# ====================================================
# 9  Trial function with weighted up–down updating
# ====================================================
def run_trial(phase, angle_bias, expect_level, mode):
    """
    Run one trial:
      - mode: "true" uses PROP directly; "medium" uses MEDIUM_PROP_MODE
      - weighted up–down staircase updates PROP on practice/true trials
    """
    # select proportion based on mode
    prop = (PROP[angle_bias][expect_level]
            if mode == "true"
            else MEDIUM_PROP_MODE[angle_bias])
    # set cue colour
    cue_col = low_col if expect_level == "low" else high_col
    fix.color = cue_col                           # fixation colour
    square.fillColor = square.lineColor = cue_col # stimuli colours
    dot.fillColor    = dot.lineColor    = cue_col

    # show fixation
    fix.draw(); win.flip(); core.wait(0.5)

    # random start positions for stimuli
    sqX, sqY = np.random.uniform(-200,200,2)
    dotX, dotY = -sqX, -sqY
    square.pos, dot.pos = (sqX, sqY), (dotX, dotY)

    # choose target shape
    target_shape = random.choice(["square","dot"])
    snip = ou_lib[random.randrange(LIB_N)]  # pick OU snippet

    # prepare for continuous movement loop
    mouse.setPos((0,0)); last = (0,0)
    vt = vd = np.zeros(2)  # velocity trackers
    clk = core.Clock(); frame = 0

    # movement loop for 3 seconds
    while clk.getTime() < 3.0:
        x,y = mouse.getPos()             # current mouse
        dx,dy = x-last[0], y-last[1]     # mouse delta
        last = (x,y)
        dx,dy = rotate(dx,dy, angle_bias)  # apply rotation bias
        ou_dx,ou_dy = snip[frame % LIB_LEN]# OU motion
        frame += 1

        # normalize OU magnitude to mouse delta magnitude
        mag_m, mag_o = math.hypot(dx,dy), math.hypot(ou_dx,ou_dy)
        if mag_o > 0:
            ou_dx,ou_dy = ou_dx/mag_o*mag_m, ou_dy/mag_o*mag_m

        # mix according to prop
        targ_dx = prop*dx + (1-prop)*ou_dx
        targ_dy = prop*dy + (1-prop)*ou_dy

        # other shape movement = OU only
        dist_dx, dist_dy = ou_dx, ou_dy

        # smooth velocities
        vt = LOWPASS*vt + (1-LOWPASS)*np.array([targ_dx, targ_dy])
        vd = LOWPASS*vd + (1-LOWPASS)*np.array([dist_dx, dist_dy])

        # apply movement to correct shape
        if target_shape == "square":
            square.pos = confine(tuple(square.pos + vt))
            dot.pos    = confine(tuple(dot.pos + vd))
        else:
            dot.pos    = confine(tuple(dot.pos + vt))
            square.pos = confine(tuple(square.pos + vd))

        square.draw(); dot.draw(); win.flip()

    # collect shape & confidence response
    msg.text = (
        "Which shape did you control & how confident?\n"
        "1 Square (guess)   2 Square (confident)\n"
        "3 Dot    (confident) 4 Dot (guess)"
    )
    msg.draw(); win.flip()
    key = event.waitKeys(keyList=CONF_KEYS+["escape"])[0]
    if key == "escape":                     # abort experiment
        _save_data_on_exit(); core.quit()
    resp_shape = SHAPE_FROM_KEY[key]        # map to shape
    conf_lvl = CONF_FROM_KEY[key]           # map to confidence
    correct = int(resp_shape == target_shape)# 1 if correct, else 0
    core.wait(0.5)                          # blank interval

    # continuous control rating
    if SIMULATE:
        rating = random.randint(0,100)      # random rating in simulate
    else:
        msg.text = "How much control? (0–100, Enter)"
        msg.draw(); win.flip()
        txt = ""; done = False
        while not done:
            for k in event.waitKeys():
                if k in ("return","enter"):
                    done = True
                elif k == "backspace":
                    txt = txt[:-1]
                elif k.isdigit() and len(txt) < 3:
                    txt += k
                elif k == "escape":
                    _save_data_on_exit(); core.quit()
            msg.text = f"How much control? (0–100)\nCurrent: {txt}"
            msg.draw(); win.flip()
        rating = float(txt) if txt else 0.0

    # staircase update only on practice & true trials
    if phase == "practice" or mode == "true":
        if correct:  # correct → make harder by stepping DOWN
            PROP[angle_bias][expect_level] = max(
                0.0,
                PROP[angle_bias][expect_level] - STEP_DOWN[expect_level]
            )
        else:        # incorrect → make easier by stepping UP
            PROP[angle_bias][expect_level] = min(
                1.0,
                PROP[angle_bias][expect_level] + STEP_UP[expect_level]
            )

    # return results dict
    return {
        "phase": phase,
        "angle_bias": angle_bias,
        "expect_level": expect_level,
        "mode": mode,
        "true_shape": target_shape,
        "resp_shape": resp_shape,
        "conf_level": conf_lvl,
        "accuracy": correct,
        "control_rating": rating,
        "prop_used": prop
    }

# ====================================================
# 10  Practice block (60 trials)
# ====================================================
PRACTICE_PER_CELL = 3                                    # 3 trials per mode×cue
conds = [(m,e) for m in MODES for e in EXPECT] * PRACTICE_PER_CELL
random.shuffle(conds)                                    # randomize order

msg.text = "PRACTICE – ≈5 min.\nPress any key."
msg.draw(); win.flip(); event.waitKeys()                 # start prompt

for i, (m,e) in enumerate(conds, start=1):
    res = run_trial("practice", m, e, "true")           # run practice trial
    for k,v in res.items():
        thisExp.addData(k, v)                           # record data
    thisExp.nextEntry()                                 # next line in file
    if i % 30 == 0 and not SIMULATE:                    # midway break
        msg.text = "Short break – press any key."
        msg.draw(); win.flip(); event.waitKeys()

# recompute medium proportions for test based on converged staircases
MEDIUM_PROP_MODE = {
    m: (PROP[m]["low"] + PROP[m]["high"]) / 2
    for m in MODES
}

# ====================================================
# 11  Test block (300 medium trials)
# ====================================================
MEDIUM_PER_CELL = 2                                     # 2 trials per mode×cue
test_conds = [(m,e) for m in MODES for e in EXPECT] * MEDIUM_PER_CELL
random.shuffle(test_conds)

msg.text = "MAIN BLOCK – ≈25 min.\nPress any key to start."
msg.draw(); win.flip(); event.waitKeys()                # start prompt

trial_counter = 0
for m,e in test_conds:
    trial_counter += 1
    res = run_trial("test", m, e, "medium")             # run test trial
    for k,v in res.items():
        thisExp.addData(k, v)                           # record data
    thisExp.nextEntry()
    if trial_counter % BREAK_EVERY == 0 and not SIMULATE:  # periodic rest
        msg.text = "Break – rest 30 s, then continue"
        msg.draw(); win.flip(); core.wait(5)

# ====================================================
# 12  Save & goodbye
# ====================================================
thisExp.saveAsWideText(csv_path)                        # final save
print(f"Data saved to {csv_path}")                      # console message

msg.text = "Thank you – task complete! Press any key."
msg.draw(); win.flip(); event.waitKeys()                # final prompt

win.close()                                             # close window
core.quit()                                             # close experiment