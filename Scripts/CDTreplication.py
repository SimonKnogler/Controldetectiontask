#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Control Detection Task Demo (20 Trials, 0° vs 90° Counterbalanced)
Using an Ornstein–Uhlenbeck process for distractor snippets.
Implements a 2-up/1-down staircase with 5% steps (aiming ~70% accuracy).

Saves exactly ONE .csv file to:
'/Users/simonknogler/Desktop/PhD/Control Detection Task/data'
"""

import os, random, numpy as np
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

# Create a full-screen window
win = visual.Window(size=[1920,1080], fullscr=True, color=[0.5,0.5,0.5], units="pix")

# Global clock
global_clock = core.Clock()

#############################################
# 2. Stimuli, Timing & Helper Functions
#############################################

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
        if np.hypot(x1, y1) <= 250:
            x2 = random.uniform(-250, 250)
            y2 = random.uniform(-250, 250)
            if np.hypot(x2, y2) <= 250:
                dist = np.hypot(x1 - x2, y1 - y2)
                if dist >= 200:
                    return (x1, y1, x2, y2)

# Define stimuli
square = visual.Rect(win, width=40, height=40, fillColor="black", lineColor="black")
dot    = visual.Circle(win, radius=20, fillColor="black", lineColor="black")
fixation = visual.TextStim(win, text="+", color="white", height=30)
instruction_text = visual.TextStim(win, text="", color="white", height=28, wrapWidth=1000)

#############################################
# 3. Pre-Experiment Free Movement Trial (30 sec)
#############################################

instruction_text.text = (
    "Pre-Experiment Trial\n\n"
    "In this trial you will see two shapes (a square and a dot) just like in the main task.\n"
    "Move your mouse naturally for 30 seconds.\n"
    "Afterward, indicate which shape you felt you controlled more.\n\n"
    "This trial will be used to adapt the distractor movement to your natural movement patterns.\n\n"
    "Press any key to start."
)
instruction_text.draw()
win.flip()
event.waitKeys()

# Pre-trial parameters
pre_trial_duration = 30.0

# Fixation period (0.5 sec)
fixation.draw()
win.flip()
core.wait(0.5)

# Randomize starting positions
sqX, sqY, dotX, dotY = random_positions()
square.pos = (sqX, sqY)
dot.pos    = (dotX, dotY)

# Randomly choose target shape for this trial
tShape = random.choice(["square", "dot"])

# Generate a dummy OU snippet for the free trial distractor movement.
# We'll generate a snippet covering the entire 30 sec (assuming 60 Hz)
pre_trial_n_frames = int(pre_trial_duration * 60)
dummy_snippet = None
# Use a basic OU process with zero bias since we haven't computed the participant's bias yet.
def generate_OU_snippet(n_frames, theta=0.2, sigma=2.0, dt=1.0, linear_bias=(0,0), angular_bias=0):
    v = np.zeros((n_frames, 2))
    for t in range(1, n_frames):
        v[t,0] = v[t-1,0] - theta * v[t-1,0] * dt + sigma * np.random.randn() * np.sqrt(dt) + linear_bias[0] * dt
        v[t,1] = v[t-1,1] - theta * v[t-1,1] * dt + sigma * np.random.randn() * np.sqrt(dt) + linear_bias[1] * dt
        speed = np.hypot(v[t,0], v[t,1])
        angle = np.arctan2(v[t,1], v[t,0])
        angle += angular_bias * dt
        v[t,0] = speed * np.cos(angle)
        v[t,1] = speed * np.sin(angle)
    return v

dummy_snippet = generate_OU_snippet(n_frames=pre_trial_n_frames, theta=0.2, sigma=2.0, dt=1.0,
                                    linear_bias=(0,0), angular_bias=0)

# Initialize momentum for distractor movement in free trial
distractor_velocity = np.array([0.0, 0.0])
momentum_coef = 0.8
# For the free trial, set a constant proportion (similar to main trial) for mixing movements.
prop_trial = 0.6

# Prepare to record free-trial movements for bias computation
pre_movements = []
mouse = event.Mouse(win=win, visible=True)
mouse.setPos((0,0))
last_mx, last_my = mouse.getPos()
pre_trial_clock = core.Clock()
pre_trial_clock.reset()
frameN = 0

while pre_trial_clock.getTime() < pre_trial_duration:
    mx, my = mouse.getPos()
    dx = mx - last_mx
    dy = my - last_my
    pre_movements.append((dx, dy))
    last_mx, last_my = mx, my

    # For target movement: mix the mouse movement with the dummy snippet's movement.
    mag_mouse = np.hypot(dx, dy)
    pre_dx, pre_dy = dummy_snippet[frameN % pre_trial_n_frames]
    mag_snippet = np.hypot(pre_dx, pre_dy)
    if mag_snippet > 0:
        norm_pre_dx = (pre_dx / mag_snippet) * mag_mouse
        norm_pre_dy = (pre_dy / mag_snippet) * mag_mouse
    else:
        norm_pre_dx, norm_pre_dy = 0, 0

    targ_dx = (1 - prop_trial) * dx + prop_trial * norm_pre_dx
    targ_dy = (1 - prop_trial) * dy + prop_trial * norm_pre_dy

    # Update distractor movement using momentum
    current_ou_velocity = np.array([norm_pre_dx, norm_pre_dy])
    distractor_velocity = momentum_coef * distractor_velocity + (1 - momentum_coef) * current_ou_velocity
    dist_dx, dist_dy = distractor_velocity

    if tShape == "square":
        square.pos = confine_position((square.pos[0] + targ_dx, square.pos[1] + targ_dy), 250)
        dot.pos    = confine_position((dot.pos[0] + dist_dx, dot.pos[1] + dist_dy), 250)
    else:
        dot.pos    = confine_position((dot.pos[0] + targ_dx, dot.pos[1] + targ_dy), 250)
        square.pos = confine_position((square.pos[0] + dist_dx, square.pos[1] + dist_dy), 250)

    square.draw()
    dot.draw()
    win.flip()
    frameN += 1

# Forced choice response after free trial
choice_text = visual.TextStim(win,
    text="Which shape did you control more?\n(S) Square   (D) Dot",
    color="white", height=30)
choice_text.draw()
win.flip()
ckey = event.waitKeys(keyList=["s", "d", "escape"])
if "escape" in ckey:
    core.quit()
pre_choice = "square" if ckey[0] == "s" else "dot"
core.wait(0.5)

# Compute bias parameters from the recorded free-trial movements
pre_movements = np.array(pre_movements)
avg_linear_bias = pre_movements.mean(axis=0)
print("Computed average linear bias:", avg_linear_bias)

angles = np.arctan2(pre_movements[:,1], pre_movements[:,0])
angular_changes = np.diff(angles)
angular_changes = (angular_changes + np.pi) % (2*np.pi) - np.pi  # adjust for wrap-around
avg_angular_bias = angular_changes.mean() if angular_changes.size > 0 else 0
print("Computed average angular bias:", avg_angular_bias)

#############################################
# 4. Generate Ornstein–Uhlenbeck Library (with bias)
#############################################

# Now generate prerecorded OU snippets for the main task using the computed biases.
def generate_OU_snippet(n_frames=300, theta=0.2, sigma=2.0, dt=1.0,
                          linear_bias=(0,0), angular_bias=0):
    v = np.zeros((n_frames, 2))
    for t in range(1, n_frames):
        v[t,0] = v[t-1,0] - theta * v[t-1,0] * dt + sigma * np.random.randn() * np.sqrt(dt) + linear_bias[0] * dt
        v[t,1] = v[t-1,1] - theta * v[t-1,1] * dt + sigma * np.random.randn() * np.sqrt(dt) + linear_bias[1] * dt
        speed = np.hypot(v[t,0], v[t,1])
        angle = np.arctan2(v[t,1], v[t,0])
        angle += angular_bias * dt
        v[t,0] = speed * np.cos(angle)
        v[t,1] = speed * np.sin(angle)
    return v

snippet_length = 300   # 5 seconds at 60Hz
num_snippets = 3000    # Reduced number of prerecorded movements
library = []
for _ in range(num_snippets):
    library.append(generate_OU_snippet(n_frames=snippet_length, theta=0.2, sigma=2.0, dt=1.0,
                                       linear_bias=avg_linear_bias, angular_bias=avg_angular_bias))
library = np.array(library)

#############################################
# 5. 2-up/1-down Staircase
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
# 6. Condition List (20 Trials, 10 at 0°, 10 at 90°)
#############################################

conditions_list = [{"angle":0}]*10 + [{"angle":90}]*10
random.shuffle(conditions_list)

#############################################
# 7. Main Task
#############################################

instruction_text.text = (
    "CONTROL DETECTION TASK (20-Trial Demo, 2-up/1-down)\n\n"
    "10 trials at 0°, 10 at 90°, randomly ordered.\n"
    "Move your mouse for 5 seconds each trial.\n"
    "One shape is partly under your control (mixed with snippet),\n"
    "the other shape is purely snippet.\n\n"
    "After each trial, choose which shape you controlled more, then rate your confidence.\n"
    "The staircase adjusts difficulty in 5% steps.\n\n"
    "Press any key to start."
)
instruction_text.draw()
win.flip()
event.waitKeys()

try:
    global_clock.reset()
    trial_count = 0

    for trial_info in conditions_list:
        trial_count += 1
        angle_bias = trial_info["angle"]

        if angle_bias == 0:
            prop_trial = prop_0deg
            rolling_acc = rolling_mean(history_0deg)
        else:
            prop_trial = prop_90deg
            rolling_acc = rolling_mean(history_90deg)

        trial_time = global_clock.getTime()

        fixation.draw()
        win.flip()
        core.wait(0.5)

        sqX, sqY, dotX, dotY = random_positions()
        square.pos = (sqX, sqY)
        dot.pos    = (dotX, dotY)

        tShape = random.choice(["square", "dot"])

        # Movement loop for 5-second trial
        snippet = library[random.randrange(library.shape[0])]
        mouse = event.Mouse(win=win, visible=False)
        mouse.setPos((0,0))
        distractor_velocity = np.array([0.0, 0.0])
        momentum_coef = 0.8
        clock = core.Clock()
        clock.reset()
        last_mx, last_my = 0, 0
        frameN = 0

        while clock.getTime() < 5.0:
            mx, my = mouse.getPos()
            dx = mx - last_mx
            dy = my - last_my
            last_mx, last_my = mx, my

            pre_dx, pre_dy = snippet[frameN % snippet_length]
            mag_mouse = np.hypot(dx, dy)
            mag_snippet = np.hypot(pre_dx, pre_dy)
            if mag_snippet > 0:
                norm_pre_dx = (pre_dx / mag_snippet) * mag_mouse
                norm_pre_dy = (pre_dy / mag_snippet) * mag_mouse
            else:
                norm_pre_dx, norm_pre_dy = 0, 0

            # In main task, the target movement is a mix of the rotated mouse movement and the OU snippet.
            # Here we mimic that process.
            dx_biased, dy_biased = rotate_vector(dx, dy, angle_bias)
            targ_dx = (1 - prop_trial) * dx_biased + prop_trial * norm_pre_dx
            targ_dy = (1 - prop_trial) * dy_biased + prop_trial * norm_pre_dy

            current_ou_velocity = np.array([norm_pre_dx, norm_pre_dy])
            distractor_velocity = momentum_coef * distractor_velocity + (1 - momentum_coef) * current_ou_velocity
            dist_dx, dist_dy = distractor_velocity

            if tShape == "square":
                square.pos = confine_position((square.pos[0] + targ_dx, square.pos[1] + targ_dy), 250)
                dot.pos = confine_position((dot.pos[0] + dist_dx, dot.pos[1] + dist_dy), 250)
            else:
                dot.pos = confine_position((dot.pos[0] + targ_dx, dot.pos[1] + targ_dy), 250)
                square.pos = confine_position((square.pos[0] + dist_dx, square.pos[1] + dist_dy), 250)

            square.draw()
            dot.draw()
            win.flip()
            frameN += 1

        choice_text = visual.TextStim(win,
            text="Which shape did you control more?\n(S) Square   (D) Dot",
            color="white", height=30)
        choice_text.draw()
        win.flip()
        ckey = event.waitKeys(keyList=["s", "d", "escape"])
        if "escape" in ckey:
            raise KeyboardInterrupt
        chosen_shape = "square" if ckey[0] == "s" else "dot"
        accuracy = 1 if (chosen_shape == tShape) else 0

        conf_text = visual.TextStim(win,
            text="Confidence?\n(L) Low   (H) High",
            color="white", height=30)
        conf_text.draw()
        win.flip()
        cckey = event.waitKeys(keyList=["l", "h", "escape"])
        if "escape" in cckey:
            raise KeyboardInterrupt
        confidence = "low" if cckey[0] == "l" else "high"

        win.flip()
        core.wait(0.5)

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

        thisExp.addData("time", trial_time)
        thisExp.addData("condition", str(angle_bias))
        thisExp.addData("accuracy", accuracy)
        thisExp.addData("prop", prop_trial)
        thisExp.addData("rolling_acc", rolling_acc)
        thisExp.addData("confidence", confidence)
        thisExp.nextEntry()

except KeyboardInterrupt:
    print("Experiment exited prematurely; saving data...")

finally:
    csv_path = os.path.join(data_path, filename + ".csv")
    thisExp.saveAsWideText(csv_path)
    print(f"Data saved to: {csv_path}")
    win.close()
    core.quit()
