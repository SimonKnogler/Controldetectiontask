#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
control_detection_task_v9_personalised.py
──────────────────────────────────────────
Identical to v9 except: every snippet is rescaled so its mean speed
(px/frame) matches the participant’s own average speed measured during
the 30-s demo.  Nothing else is changed.
"""

# ── Imports & SIMULATE toggle ──────────────────────────
import os, sys, math, random, pathlib, datetime, atexit, hashlib
import numpy as np
from psychopy import visual, event, core, data, gui
SIMULATE = False

# ── Auto-save handler ──────────────────────────────────
_saved = False
def _save():                                # save CSV once
    global _saved
    if not _saved:
        thisExp.saveAsWideText(csv_path)
        print("Data auto-saved →", csv_path)
        _saved = True
atexit.register(_save)

# ── Participant dialog & colour counterbalance ────────
expName = "ControlDetection_v9p"
expInfo = {"participant": "", "session": "001"}
if not SIMULATE and not gui.DlgFromDict(expInfo, title=expName).OK:
    core.quit()
if SIMULATE:
    expInfo["participant"] = "SIM"

low_col, high_col = random.choice([("blue", "green"), ("green", "blue")])
expInfo["low_precision_colour"]  = low_col
expInfo["high_precision_colour"] = high_col

# ── RNG & motion library ───────────────────────────────
seed = int(hashlib.sha256(expInfo["participant"].encode()).hexdigest(), 16) & 0xFFFFFFFF
rng  = np.random.default_rng(seed)

SNIP_LEN = 300
motion_pool = np.load("master_snippets.npy")           # (N,300,2)

# Pre-compute each snippet’s native mean speed (px/frame)
snippet_speeds = np.linalg.norm(motion_pool, axis=2).mean(axis=1)  # (N,)

# Placeholder for personalised speed (set after demo)
avg_demo_speed = None

# Original sampler (used only in demo)
def sample_snippet():
    idx = int(rng.integers(len(motion_pool)))
    return motion_pool[idx].astype(np.float32).T, idx  # (2,300), index

# Personalised sampler (used in trials)
def sample_snippet_scaled():
    idx = int(rng.integers(len(motion_pool)))
    snippet = motion_pool[idx].astype(np.float32)       # (300,2)
    native = snippet_speeds[idx] + 1e-9                # avoid zero
    scale  = avg_demo_speed / native
    snippet *= scale
    return snippet.T, idx                              # (2,300), index

# ── Paths & ExperimentHandler ─────────────────────────
root = pathlib.Path.cwd() / "data"; root.mkdir(exist_ok=True)
file_stem = (f"{expName}_P{expInfo['participant']}_S{expInfo['session']}_"
             f"{datetime.datetime.now():%Y%m%dT%H%M%S}")
csv_path = root / f"{file_stem}.csv"
thisExp = data.ExperimentHandler(name=expName, extraInfo=expInfo,
                                 savePickle=False, saveWideText=False,
                                 dataFileName=str(root / file_stem))

# ── Window & stimuli ──────────────────────────────────
win = visual.Window(size=(1920,1080), fullscr=not SIMULATE,
                    color=[0.5]*3, units="pix")
square = visual.Rect(win, 40, 40, fillColor="black", lineColor="black")
dot    = visual.Circle(win, 20, fillColor="black", lineColor="black")
fix    = visual.TextStim(win, text="+", color="white", height=30)
msg    = visual.TextStim(win, text="", color="white", height=26, wrapWidth=1000)

# ── Helpers ───────────────────────────────────────────
confine = lambda p,l=250: p if (r:=math.hypot(*p))<=l else (p[0]*l/r, p[1]*l/r)
rotate  = lambda vx,vy,a:(vx*math.cos(math.radians(a))-vy*math.sin(math.radians(a)),
                          vx*math.sin(math.radians(a))+vy*math.cos(math.radians(a)))

# ── Easy 30-s demo (measures avg speed) ───────────────
def demo():
    global avg_demo_speed
    prop_demo = 0.80
    msg.text = ("30-s DEMO – one shape mostly follows your mouse.\n"
                "Observe, then practice starts.\n\nPress any key.")
    msg.draw(); win.flip(); event.waitKeys()
    fix.draw(); win.flip(); core.wait(0.5)

    sqX, sqY = np.random.uniform(-200,200,2)
    square.pos, dot.pos = (sqX,sqY), (-sqX,-sqY)
    target = "square"
    (ou_x,ou_y), _ = sample_snippet()
    mouse = event.Mouse(win=win, visible=not SIMULATE); mouse.setPos((0,0))
    last = (0,0); vt = vd = np.zeros(2); frame = 0; clk = core.Clock()
    speeds = []                                 # store |Δ| each frame

    while clk.getTime() < 30:
        x,y = mouse.getPos(); dx,dy = x-last[0], y-last[1]; last = (x,y)
        speeds.append(math.hypot(dx,dy))        # collect speed
        ou_dx,ou_dy = ou_x[frame%SNIP_LEN], ou_y[frame%SNIP_LEN]; frame += 1
        mag_m,mag_o = math.hypot(dx,dy), math.hypot(ou_dx,ou_dy)
        if mag_o>0: ou_dx,ou_dy = ou_dx/mag_o*mag_m, ou_dy/mag_o*mag_m
        tdx=prop_demo*dx+(1-prop_demo)*ou_dx; tdy=prop_demo*dy+(1-prop_demo)*ou_dy
        vt = 0.8*vt + 0.2*np.array([tdx,tdy]); vd = 0.8*vd + 0.2*np.array([ou_dx,ou_dy])
        square.pos = confine(tuple(square.pos+vt)); dot.pos = confine(tuple(dot.pos+vd))
        square.draw(); dot.draw(); win.flip()

    avg_demo_speed = float(np.mean(speeds))
    print("Avg demo speed =", avg_demo_speed, "px/frame")
demo()

# ── Staircase params (unchanged) ───────────────────────
MODES=[0,90]; EXPECT=["low","high"]
TARGET={"low":0.60,"high":0.80}; BASE=0.05
STEP_DOWN={e:BASE for e in EXPECT}
STEP_UP={e:(TARGET[e]/(1-TARGET[e]))*BASE for e in EXPECT}
PROP={m:{"low":0.50,"high":0.80} for m in MODES}
MED_PROP={m:(PROP[m]["low"]+PROP[m]["high"])/2 for m in MODES}
BREAK_EVERY=3; LOWPASS=0.8
CONF_KEYS=["1","2","3","4"]
SHAPE_FROM_KEY={"1":"square","2":"square","3":"dot","4":"dot"}
CONF_FROM_KEY={k:i+1 for i,k in enumerate(CONF_KEYS)}

# ── Trial function with personalised snippet & slider ─
def run_trial(phase, angle_bias, expect_level, mode):
    prop = PROP[angle_bias][expect_level] if mode=="true" else MED_PROP[angle_bias]
    cue  = low_col if expect_level=="low" else high_col
    fix.color = square.fillColor = square.lineColor = dot.fillColor = dot.lineColor = cue
    fix.draw(); win.flip(); core.wait(0.5)

    sqX,sqY = np.random.uniform(-200,200,2)
    square.pos,dot.pos = (sqX,sqY),(-sqX,-sqY)
    target = random.choice(["square","dot"])
    (ou_x,ou_y), idx = sample_snippet_scaled(); thisExp.addData("snippet_id", idx)

    mouse = event.Mouse(win=win, visible=not SIMULATE); mouse.setPos((0,0))
    last=(0,0); vt=vd=np.zeros(2); frame=0; clk=core.Clock()
    while clk.getTime()<3:
        x,y = mouse.getPos(); dx,dy = x-last[0], y-last[1]; last=(x,y)
        dx,dy = rotate(dx,dy,angle_bias)
        ou_dx,ou_dy = ou_x[frame%SNIP_LEN], ou_y[frame%SNIP_LEN]; frame += 1
        mag_m,mag_o = math.hypot(dx,dy), math.hypot(ou_dx,ou_dy)
        if mag_o>0: ou_dx,ou_dy = ou_dx/mag_o*mag_m, ou_dy/mag_o*mag_m
        tdx = prop*dx+(1-prop)*ou_dx; tdy = prop*dy+(1-prop)*ou_dy
        vt = LOWPASS*vt + (1-LOWPASS)*np.array([tdx,tdy])
        vd = LOWPASS*vd + (1-LOWPASS)*np.array([ou_dx,ou_dy])
        if target=="square":
            square.pos = confine(tuple(square.pos+vt)); dot.pos = confine(tuple(dot.pos+vd))
        else:
            dot.pos = confine(tuple(dot.pos+vt)); square.pos = confine(tuple(square.pos+vd))
        square.draw(); dot.draw(); win.flip()

    # shape/confidence query
    msg.text=("Which shape did you control & how confident?\n"
              "1 Square (guess)  2 Square (conf)  3 Dot (conf)  4 Dot (guess)")
    msg.draw(); win.flip()
    key = event.waitKeys(keyList=CONF_KEYS+["escape"])[0]
    if key=="escape": _save(); core.quit()
    resp_shape = SHAPE_FROM_KEY[key]; conf_lvl = CONF_FROM_KEY[key]
    correct = int(resp_shape==target); core.wait(0.3)

    # Agency slider 0–100
    slider = visual.Slider(win, pos=(0,-250), size=(600,40),
                           ticks=(0,100), labels=("0","100"),
                           granularity=1, style='rating',
                           labelHeight=24, color='white', fillColor='white')
    msg.text="How much control did you feel?"; msg.draw(); slider.draw(); win.flip()
    while slider.rating is None:
        slider.draw(); msg.draw(); win.flip()
        if event.getKeys(["escape"]): _save(); core.quit()
    rating = float(slider.rating)
    core.wait(0.2)

    # Staircase update
    if phase=="practice" or mode=="true":
        if correct:
            PROP[angle_bias][expect_level] = max(
                0.0, PROP[angle_bias][expect_level]-STEP_DOWN[expect_level])
        else:
            PROP[angle_bias][expect_level] = min(
                1.0, PROP[angle_bias][expect_level]+STEP_UP[expect_level])

    return {"phase":phase,"angle_bias":angle_bias,"expect_level":expect_level,
            "mode":mode,"true_shape":target,"resp_shape":resp_shape,
            "conf_level":conf_lvl,"accuracy":correct,"control_rating":rating,
            "prop_used":prop}

# Practice block --------------------------------------
PPC=3
practice=[(m,e) for m in MODES for e in EXPECT]*PPC
random.shuffle(practice)
msg.text="PRACTICE – Press any key."; msg.draw(); win.flip(); event.waitKeys()
for i,(m,e) in enumerate(practice,1):
    res=run_trial("practice",m,e,"true")
    for k,v in res.items(): thisExp.addData(k,v); thisExp.nextEntry()

MED_PROP = {m:(PROP[m]["low"]+PROP[m]["high"])/2 for m in MODES}

# Test block ------------------------------------------
MPC=2
test=[(m,e) for m in MODES for e in EXPECT]*MPC
random.shuffle(test)
msg.text="MAIN BLOCK – Press any key."; msg.draw(); win.flip(); event.waitKeys()
t=0
for m,e in test:
    t+=1
    res=run_trial("test",m,e,"medium")
    for k,v in res.items(): thisExp.addData(k,v); thisExp.nextEntry()
    if t%BREAK_EVERY==0 and not SIMULATE:
        msg.text="Break – press any key"; msg.draw(); win.flip(); event.waitKeys()

# Save & exit -----------------------------------------
thisExp.saveAsWideText(csv_path); print("Saved →",csv_path)
msg.text="Thanks – finished!"; msg.draw(); win.flip(); event.waitKeys()
win.close(); core.quit()