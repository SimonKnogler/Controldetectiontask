#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Control Detection Task Demo (20 Trials, 0° vs 90° Counterbalanced)
Using an Ornstein–Uhlenbeck process for distractor snippets.
Implements a 2-up/1-down staircase with 5% steps (aiming ~70% accuracy).

**Saves exactly ONE .csv file** (no .log, no .pkl) to:
'/Users/simonknogler/Desktop/PhD/Control Detection Task/data'

Columns per trial (one row each):
  time, condition, accuracy, prop, rolling_acc, confidence
"""

import os, sys, random, numpy as np
from psychopy import visual, core, data, event, gui

#############################################
# 1. Experiment Setup
#############################################

expName = "ControlDetection_2up1down_Columns"
expInfo = {"participant": "", "session": "001"}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()

filename = f"{expName}_P{expInfo['participant']}_S{expInfo['session']}_{data.getDateStr()}"
data_path = "/Users/simonknogler/Desktop/PhD/Controldetectiontask/data"
if not os.path.isdir(data_path):
    os.makedirs(data_path)

# Create an ExperimentHandler that won't auto-save .log or .pkl
thisExp = data.ExperimentHandler(
    name=expName,
    extraInfo=expInfo,
    savePickle=False,    # do not save .pkl
    saveWideText=False,  # do not auto-save .csv
    dataFileName=os.path.join(data_path, filename)
)

win = visual.Window(size=[1920,1080], fullscr=False, color=[0.5,0.5,0.5], units="pix")

# Global clock
global_clock = core.Clock()

#############################################
# 2. Generate Ornstein–Uhlenbeck Library
#############################################

def generate_OU_snippet(n_frames=300, theta=0.2, sigma=2.0, dt=1.0):
    """Generate snippet (n_frames x 2) using an OU process for velocity."""
    v = np.zeros((n_frames, 2))
    for t in range(1, n_frames):
        v[t,0] = v[t-1,0] - theta*v[t-1,0]*dt + sigma*np.random.randn()*np.sqrt(dt)
        v[t,1] = v[t-1,1] - theta*v[t-1,1]*dt + sigma*np.random.randn()*np.sqrt(dt)
    return v

snippet_length = 300  # 5s @ 60Hz
num_snippets = 2000
library = []
for _ in range(num_snippets):
    library.append(generate_OU_snippet())
library = np.array(library)

def get_random_snippet():
    idx = random.randrange(library.shape[0])
    return library[idx]

#############################################
# 3. Stimuli, Timing & Helper Functions
#############################################

trial_duration = 5.0
feedback_isi   = 0.5

square = visual.Rect(win, width=40, height=40, fillColor="black", lineColor="black")
dot    = visual.Circle(win, radius=20, fillColor="black", lineColor="black")
fixation = visual.TextStim(win, text="+", color="white", height=30)
instruction_text = visual.TextStim(win, text="", color="white", height=28, wrapWidth=1000)

def confine_position(pos, limit=250):
    x, y = pos
    x = max(-limit, min(x, limit))
    y = max(-limit, min(y, limit))
    return (x, y)

def rotate_vector(x, y, angle_deg):
    theta = np.deg2rad(angle_deg)
    cosT  = np.cos(theta)
    sinT  = np.sin(theta)
    return (x*cosT - y*sinT, x*sinT + y*cosT)

def random_positions():
    while True:
        x1 = random.uniform(-250, 250)
        y1 = random.uniform(-250, 250)
        if (x1**2 + y1**2)**0.5 <= 250:
            x2 = random.uniform(-250, 250)
            y2 = random.uniform(-250, 250)
            if (x2**2 + y2**2)**0.5 <= 250:
                dist = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                if dist >= 200:
                    return (x1, y1, x2, y2)

#############################################
# 4. 2-up/1-down Staircase
#############################################
prop_0deg = 0.60
prop_90deg = 0.45
step_size = 0.05
min_prop  = 0.0
max_prop  = 1.0

consec_correct_0deg = 0
consec_correct_90deg = 0

history_0deg = []
history_90deg = []

def rolling_mean(arr, window=10):
    if len(arr) == 0:
        return 0.0
    return np.mean(arr[-window:])

#############################################
# 5. Condition List (20 Trials, 10 at 0°, 10 at 90°)
#############################################
conditions_list = [{"angle":0}]*10 + [{"angle":90}]*10
random.shuffle(conditions_list)

shapes = ["square","dot"]

#############################################
# 6. Instructions
#############################################
instruction_text.text = (
    "CONTROL DETECTION TASK (20-Trial Demo, 2-up/1-down)\n\n"
    "10 trials at 0°, 10 at 90°, randomly ordered.\n"
    "Move your mouse for 5s each trial.\n"
    "One shape is partly under your control (mixed with snippet),\n"
    "the other shape is purely snippet.\n\n"
    "After each trial, choose which shape you controlled more, then rate your confidence.\n"
    "The staircase adjusts difficulty in 5% steps.\n\n"
    "Press any key to start."
)
instruction_text.draw()
win.flip()
event.waitKeys()

#############################################
# 7. Main Task (try-finally to ensure CSV is saved if user quits)
#############################################

try:
    global_clock.reset()
    trial_count = 0

    for trial_info in conditions_list:
        trial_count += 1
        angle_bias = trial_info["angle"]

        # Current proportion before updating
        if angle_bias == 0:
            prop_trial = prop_0deg
            rolling_acc = rolling_mean(history_0deg)
        else:
            prop_trial = prop_90deg
            rolling_acc = rolling_mean(history_90deg)

        # Trial start time
        trial_time = global_clock.getTime()

        # Fixation
        fixation.draw()
        win.flip()
        core.wait(0.5)

        # Random positions
        sqX, sqY, dotX, dotY = random_positions()
        square.pos = (sqX, sqY)
        dot.pos    = (dotX, dotY)

        # Randomly pick target shape
        tShape = random.choice(shapes)

        # Movement loop
        snippet = get_random_snippet()
        mouse = event.Mouse(win=win, visible=False)
        mouse.setPos((0,0))
        clock = core.Clock()
        clock.reset()
        last_mx, last_my = 0, 0
        frameN = 0

        while clock.getTime() < trial_duration:
            mx, my = mouse.getPos()
            dx = mx - last_mx
            dy = my - last_my
            last_mx, last_my = mx, my

            pre_dx, pre_dy = snippet[frameN % snippet_length]
            mag_mouse   = (dx**2 + dy**2)**0.5
            mag_snippet = (pre_dx**2 + pre_dy**2)**0.5

            if mag_snippet > 0:
                norm_pre_dx = (pre_dx / mag_snippet) * mag_mouse
                norm_pre_dy = (pre_dy / mag_snippet) * mag_mouse
            else:
                norm_pre_dx, norm_pre_dy = 0, 0

            dx_biased, dy_biased = rotate_vector(dx, dy, angle_bias)

            # Target
            targ_dx = (1 - prop_trial)*dx_biased + prop_trial*norm_pre_dx
            targ_dy = (1 - prop_trial)*dy_biased + prop_trial*norm_pre_dy
            # Distractor
            dist_dx = norm_pre_dx
            dist_dy = norm_pre_dy

            if tShape == "square":
                square.pos = confine_position((square.pos[0] + targ_dx, square.pos[1] + targ_dy), 250)
                dot.pos    = confine_position((dot.pos[0] + dist_dx,  dot.pos[1] + dist_dy), 250)
            else:
                dot.pos    = confine_position((dot.pos[0] + targ_dx,  dot.pos[1] + targ_dy), 250)
                square.pos = confine_position((square.pos[0] + dist_dx, square.pos[1] + dist_dy), 250)

            square.draw()
            dot.draw()
            win.flip()
            frameN += 1

        # Forced choice
        choice_text = visual.TextStim(win,
            text="Which shape did you control more?\n(S) Square   (D) Dot",
            color="white", height=30)
        choice_text.draw()
        win.flip()
        ckey = event.waitKeys(keyList=["s","d","escape"])
        if "escape" in ckey:
            raise KeyboardInterrupt
        chosen_shape = "square" if ckey[0] == "s" else "dot"
        accuracy = 1 if (chosen_shape == tShape) else 0

        # Confidence
        conf_text = visual.TextStim(win,
            text="Confidence?\n(L) Low   (H) High",
            color="white", height=30)
        conf_text.draw()
        win.flip()
        cckey = event.waitKeys(keyList=["l","h","escape"])
        if "escape" in cckey:
            raise KeyboardInterrupt
        confidence = "low" if cckey[0] == "l" else "high"

        win.flip()
        core.wait(feedback_isi)

        # Staircase update (2-up/1-down)
        if angle_bias == 0:
            if accuracy == 1:
                consec_correct_0deg += 1
                if consec_correct_0deg == 2:
                    prop_0deg = min(prop_0deg + step_size, max_prop)
                    consec_correct_0deg = 0
            else:
                prop_0deg = max(prop_0deg - step_size, min_prop)
                consec_correct_0deg = 0
            history_0deg.append(accuracy)
        else:
            if accuracy == 1:
                consec_correct_90deg += 1
                if consec_correct_90deg == 2:
                    prop_90deg = min(prop_90deg + step_size, max_prop)
                    consec_correct_90deg = 0
            else:
                prop_90deg = max(prop_90deg - step_size, min_prop)
                consec_correct_90deg = 0
            history_90deg.append(accuracy)

        # Save columns
        thisExp.addData("time", trial_time)
        thisExp.addData("condition", str(angle_bias))  # "0" or "90"
        thisExp.addData("accuracy", accuracy)          # 1 or 0
        thisExp.addData("prop", prop_trial)            # proportion used
        thisExp.addData("rolling_acc", rolling_acc)    # rolling accuracy prior to this trial
        thisExp.addData("confidence", confidence)      # "low" or "high"
        thisExp.nextEntry()

except KeyboardInterrupt:
    print("Experiment exited prematurely; saving data...")

finally:
    # Save the single CSV
    csv_path = os.path.join(data_path, filename + ".csv")
    thisExp.saveAsWideText(csv_path)
    print(f"Data saved to: {csv_path}")
    win.close()
    core.quit()
