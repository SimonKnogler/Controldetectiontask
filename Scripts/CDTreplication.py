#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Control Detection Task with Two Staircases:
 - "Blue" color cue ~60% accuracy
 - "Green" color cue ~90% accuracy
 - After staircases converge, use their average as "medium" in the test phase
 - Combined (control + confidence) 4-point response, plus 0–100 control rating
 - Half of all trials at 0°, half at 90° angular bias
"""

import os, random, numpy as np
from psychopy import visual, core, data, event, gui

#############################################
# 1. Experiment Setup
#############################################

expName = "ControlDetection_TwoStaircases"
expInfo = {"participant": "", "session": "001"}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()

filename = f"{expName}_P{expInfo['participant']}_S{expInfo['session']}_{data.getDateStr()}"
data_path = "/Users/simonknogler/Desktop/PhD/Controldetectiontask/data"
if not os.path.isdir(data_path):
    os.makedirs(data_path)

thisExp = data.ExperimentHandler(
    name=expName,
    extraInfo=expInfo,
    savePickle=False,
    saveWideText=False,
    dataFileName=os.path.join(data_path, filename)
)

win = visual.Window(size=[1920,1080], fullscr=True, color=[0.5,0.5,0.5], units="pix")
global_clock = core.Clock()

#############################################
# 2. Stimuli & Helper Functions
#############################################

def confine_position(pos, limit=250):
    """Keep (x,y) within a circle of radius=limit."""
    x, y = pos
    x = max(-limit, min(x, limit))
    y = max(-limit, min(y, limit))
    return (x, y)

def rotate_vector(x, y, angle_deg):
    """Rotate (x,y) by angle_deg."""
    theta = np.deg2rad(angle_deg)
    cosT  = np.cos(theta)
    sinT  = np.sin(theta)
    return (x*cosT - y*sinT, x*sinT + y*cosT)

def random_positions():
    """Random non-overlapping (x,y) for square & dot within circle radius=250."""
    while True:
        x1 = random.uniform(-250, 250)
        y1 = random.uniform(-250, 250)
        if np.hypot(x1, y1) <= 250:
            x2 = random.uniform(-250, 250)
            y2 = random.uniform(-250, 250)
            if np.hypot(x2, y2) <= 250:
                dist = np.hypot(x1 - x2, y1 - y2)
                if dist >= 200:  # ensure they don't start too close
                    return (x1, y1, x2, y2)

square   = visual.Rect(win, width=40, height=40, fillColor="black", lineColor="black")
dot      = visual.Circle(win, radius=20, fillColor="black", lineColor="black")
fixation = visual.TextStim(win, text="+", color="white", height=30)
instruction_text = visual.TextStim(win, text="", color="white", height=28, wrapWidth=1000)

def generate_OU_snippet(n_frames=300, theta=0.2, sigma=2.0, dt=1.0,
                        linear_bias=(0,0), angular_bias=0):
    """
    Generate a 2D Ornstein-Uhlenbeck snippet with possible linear/ angular bias.
    Returns an array shape (n_frames, 2).
    """
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

#############################################
# 3. Optional Free Movement Trial (to estimate bias)
#############################################

instruction_text.text = (
    "Free-Movement Demo\n\n"
    "For ~30s, move your mouse naturally.\n"
    "One shape will partially follow your movement, the other won't.\n\n"
    "Press any key to start."
)
instruction_text.draw()
win.flip()
event.waitKeys()

pre_trial_duration = 30.0
fixation.draw()
win.flip()
core.wait(0.5)

sqX, sqY, dotX, dotY = random_positions()
square.pos = (sqX, sqY)
dot.pos    = (dotX, dotY)

tShape = random.choice(["square", "dot"])
pre_trial_n_frames = int(pre_trial_duration * 60)
dummy_snippet = generate_OU_snippet(n_frames=pre_trial_n_frames, theta=0.2, sigma=2.0, dt=1.0)

distractor_velocity = np.array([0.0, 0.0])
momentum_coef = 0.8
prop_demo = 0.6  # proportion of snippet vs. direct mouse

mouse = event.Mouse(win=win, visible=True)
mouse.setPos((0,0))
last_mx, last_my = mouse.getPos()
pre_movements = []
pre_trial_clock = core.Clock()
pre_trial_clock.reset()
frameN = 0

while pre_trial_clock.getTime() < pre_trial_duration:
    mx, my = mouse.getPos()
    dx = mx - last_mx
    dy = my - last_my
    pre_movements.append((dx, dy))
    last_mx, last_my = mx, my

    ou_dx, ou_dy = dummy_snippet[frameN % pre_trial_n_frames]
    mag_mouse = np.hypot(dx, dy)
    mag_snippet = np.hypot(ou_dx, ou_dy)
    if mag_snippet > 0:
        norm_ou_dx = (ou_dx / mag_snippet) * mag_mouse
        norm_ou_dy = (ou_dy / mag_snippet) * mag_mouse
    else:
        norm_ou_dx, norm_ou_dy = 0, 0

    targ_dx = (1 - prop_demo) * dx + prop_demo * norm_ou_dx
    targ_dy = (1 - prop_demo) * dy + prop_demo * norm_ou_dy

    current_ou_velocity = np.array([norm_ou_dx, norm_ou_dy])
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

choice_text = visual.TextStim(win,
    text="Which shape did you control more?\n(S) Square   (D) Dot",
    color="white", height=30)
choice_text.draw()
win.flip()
ckey = event.waitKeys(keyList=["s","d","escape"])
if "escape" in ckey:
    core.quit()
pre_choice = "square" if ckey[0] == "s" else "dot"

# Estimate linear & angular biases
pre_movements = np.array(pre_movements)
avg_linear_bias = pre_movements.mean(axis=0)

angles = np.arctan2(pre_movements[:,1], pre_movements[:,0])
angular_changes = np.diff(angles)
angular_changes = (angular_changes + np.pi) % (2*np.pi) - np.pi
avg_angular_bias = angular_changes.mean() if angular_changes.size>0 else 0

#############################################
# 4. Generate a Library of OU Snippets with Bias
#############################################

snippet_length = 300
num_snippets = 2000
ou_library = []
for _ in range(num_snippets):
    snippet = generate_OU_snippet(
        n_frames=snippet_length,
        theta=0.2,
        sigma=2.0,
        dt=1.0,
        linear_bias=avg_linear_bias,
        angular_bias=avg_angular_bias
    )
    ou_library.append(snippet)
ou_library = np.array(ou_library)  # shape (num_snippets, snippet_length, 2)

#############################################
# 5. Two Separate Staircases for Colors:
#    Blue -> ~60% accuracy
#    Green -> ~90% accuracy
#
# We'll do 10 practice trials total:
#  5 at angle=0°, 5 at angle=90°, each randomly assigned color=blue or color=green
#############################################

# Start each color's "proportion" at some mid-range
blue_prop = 0.50
green_prop = 0.70

# We'll track performance for each color so we can roughly converge to 60% or 90%.
# For a *simple* approach, after each trial, we look at the running accuracy so far.
# If running accuracy is above the target, we reduce prop a little; if below, we increase prop.
# (You could also do a 3-down/1-up or 1-down/3-up approach. The below is just a demonstration.)

blue_correct = 0
blue_total   = 0
green_correct = 0
green_total   = 0

TARGET_BLUE  = 0.60  # ~60%
TARGET_GREEN = 0.90  # ~90%
STEP_SIZE    = 0.05  # e.g. 5% steps

# Create 10 practice trials (5 with angle=0°, 5 with angle=90°)
angles_practice = [0]*5 + [90]*5
random.shuffle(angles_practice)

# Randomly assign half the practice trials to "blue" and half to "green"
colors_practice = ["blue"]*5 + ["green"]*5
random.shuffle(colors_practice)

practice_conditions = list(zip(angles_practice, colors_practice))
random.shuffle(practice_conditions)

instruction_text.text = (
    "PRACTICE PHASE (10 trials):\n"
    " - We'll present color cues: blue (~60% accuracy) or green (~90% accuracy)\n"
    " - Each trial also has angle 0° or 90°.\n"
    " - After the 10 trials, we'll estimate a proportion for each color.\n\n"
    "You'll do a combined control+confidence response on a 4-point scale, then a 0–100 rating.\n\n"
    "Press any key to start."
)
instruction_text.draw()
win.flip()
event.waitKeys()

def run_trial(angle_bias, color_cue, blue_prop, green_prop):
    """
    Runs a single trial using the color's current proportion (blue_prop or green_prop).
    Returns (accuracy, chosen_color, chosen_conf, rating_value).
    """
    # Fixation cross in the color
    if color_cue == "blue":
        fixation.color = "blue"
        prop = blue_prop
    else:
        fixation.color = "green"
        prop = green_prop

    fixation.draw()
    win.flip()
    core.wait(0.5)

    # Random positions for shapes
    sqX, sqY, dotX, dotY = random_positions()
    square.pos = (sqX, sqY)
    dot.pos    = (dotX, dotY)

    # Decide which shape is partially controlled
    tShape = random.choice(["square", "dot"])

    # Movement loop (~5s)
    snippet_index = random.randrange(ou_library.shape[0])
    snippet = ou_library[snippet_index]
    mouse = event.Mouse(win=win, visible=False)
    mouse.setPos((0,0))
    distractor_velocity = np.array([0.0, 0.0])
    last_mx, last_my = 0, 0
    clock = core.Clock()
    clock.reset()
    frameN = 0

    while clock.getTime() < 5.0:
        mx, my = mouse.getPos()
        dx = mx - last_mx
        dy = my - last_my
        last_mx, last_my = mx, my

        ou_dx, ou_dy = snippet[frameN % snippet_length]
        mag_mouse = np.hypot(dx, dy)
        mag_snippet = np.hypot(ou_dx, ou_dy)
        if mag_snippet > 0:
            norm_ou_dx = (ou_dx / mag_snippet) * mag_mouse
            norm_ou_dy = (ou_dy / mag_snippet) * mag_mouse
        else:
            norm_ou_dx, norm_ou_dy = 0, 0

        # Rotate the mouse movement
        dx_biased, dy_biased = rotate_vector(dx, dy, angle_bias)
        # Weighted sum
        targ_dx = (1 - prop) * dx_biased + prop * norm_ou_dx
        targ_dy = (1 - prop) * dy_biased + prop * norm_ou_dy

        momentum_coef = 0.8
        current_ou_velocity = np.array([norm_ou_dx, norm_ou_dy])
        distractor_velocity = momentum_coef * distractor_velocity + (1 - momentum_coef) * current_ou_velocity
        dist_dx, dist_dy = distractor_velocity

        if tShape == "square":
            square.pos = confine_position((square.pos[0] + targ_dx, square.pos[1] + targ_dy))
            dot.pos    = confine_position((dot.pos[0] + dist_dx, dot.pos[1] + dist_dy))
        else:
            dot.pos    = confine_position((dot.pos[0] + targ_dx, dot.pos[1] + targ_dy))
            square.pos = confine_position((square.pos[0] + dist_dx, square.pos[1] + dist_dy))

        square.draw()
        dot.draw()
        win.flip()
        frameN += 1

    # Combined control + confidence on 4-point scale
    prompt_text = (
        "Which shape did you control, and how confident are you?\n"
        "1 = Square (Guess)\n"
        "2 = Square (Confident)\n"
        "3 = Dot    (Guess)\n"
        "4 = Dot    (Confident)"
    )
    prompt_stim = visual.TextStim(win, text=prompt_text, color="white", height=26)
    prompt_stim.draw()
    win.flip()

    resp_key = event.waitKeys(keyList=["1","2","3","4","escape"])
    if "escape" in resp_key:
        core.quit()

    choice_map = {
        "1": ("square","guess"),
        "2": ("square","confident"),
        "3": ("dot","guess"),
        "4": ("dot","confident")
    }
    chosen_shape, chosen_conf = choice_map[resp_key[0]]
    accuracy = 1 if (chosen_shape == tShape) else 0

    # 0–100 rating
    rating_text = "How much control did you feel (0–100)?"
    rating_stim = visual.TextStim(win, text=rating_text, color="white", height=28)
    rating_stim.draw()
    win.flip()

    rating_string = ""
    finished_rating = False
    while not finished_rating:
        keys = event.waitKeys()
        for k in keys:
            if k in ["return","enter"]:
                finished_rating = True
                break
            elif k == "backspace":
                rating_string = rating_string[:-1]
            elif k.isdigit():
                if len(rating_string) < 3:
                    rating_string += k
            elif k == "escape":
                core.quit()

        display_text = rating_text + f"\nCurrent: {rating_string}"
        rating_stim.text = display_text
        rating_stim.draw()
        win.flip()

    rating_value = float(rating_string) if rating_string else 0
    rating_value = max(0, min(100, rating_value))

    # short pause
    win.flip()
    core.wait(0.3)

    return accuracy, chosen_shape, chosen_conf, rating_value, tShape

# ---- RUN PRACTICE TRIALS ----
for angle_bias, color_cue in practice_conditions:
    # Grab the current proportion for each color
    if color_cue == "blue":
        current_prop = blue_prop
    else:
        current_prop = green_prop

    accuracy, chosen_shape, chosen_conf, rating_value, true_shape = run_trial(
        angle_bias, color_cue, blue_prop, green_prop
    )

    # Update staircases: we measure overall accuracy for each color
    if color_cue == "blue":
        blue_correct += accuracy
        blue_total   += 1
        # quick 'step' approach: if running accuracy > target, reduce; else increase
        current_accuracy = blue_correct / float(blue_total)
        if current_accuracy > TARGET_BLUE:
            blue_prop = max(0.0, blue_prop - STEP_SIZE)
        else:
            blue_prop = min(1.0, blue_prop + STEP_SIZE)
    else:
        green_correct += accuracy
        green_total   += 1
        current_accuracy = green_correct / float(green_total)
        if current_accuracy > TARGET_GREEN:
            green_prop = max(0.0, green_prop - STEP_SIZE)
        else:
            green_prop = min(1.0, green_prop + STEP_SIZE)

    # Log practice trial
    thisExp.addData("phase", "practice")
    thisExp.addData("angle_bias", angle_bias)
    thisExp.addData("color_cue", color_cue)
    thisExp.addData("true_shape", true_shape)
    thisExp.addData("chosen_shape", chosen_shape)
    thisExp.addData("chosen_conf", chosen_conf)
    thisExp.addData("control_rating_0to100", rating_value)
    thisExp.addData("accuracy", accuracy)
    # Store the proportion used on this trial
    thisExp.addData("prop_used", current_prop)
    thisExp.addData("current_accuracy_for_color", current_accuracy)
    thisExp.nextEntry()

# Final proportions from practice
blue_final  = blue_prop
green_final = green_prop
print(f"Blue final proportion ~ {blue_final} (aiming ~60%)")
print(f"Green final proportion ~ {green_final} (aiming ~90%)")

# The "medium" level is the average
medium_prop = (blue_final + green_final)/2.0
print(f"Medium proportion = {medium_prop}")

#############################################
# 6. TEST PHASE
#    - Half the time, color is 'real' (blue_final/green_final)
#    - Half the time, color is 'medium'
#    - Also half at angle=0°, half at angle=90°
#############################################

instruction_text.text = (
    "TEST PHASE:\n"
    "Now each color cue (blue ~60%, green ~90%) is sometimes real,\n"
    "sometimes replaced with a 'medium' proportion.\n"
    "Half of the trials remain consistent with training, half are at the medium level.\n\n"
    "Press any key to continue."
)
instruction_text.draw()
win.flip()
event.waitKeys()

# For example, do 16 test trials total (8 color=blue, 8 color=green).
# Among those 8 for each color: 4 use the "true" proportion, 4 use the "medium."
# Also ensure half 0° / half 90°.
test_conditions = []
num_each_color = 8  # total number of trials for each color
num_true = 4        # how many of those use the 'real' proportion
num_medium = 4      # how many of those use the 'medium' proportion

angles_0 = [0]*(num_each_color//2)
angles_90 = [90]*(num_each_color//2)
angle_list = angles_0 + angles_90
random.shuffle(angle_list)

test_blue = [{"color":"blue","true_or_med":"true"}]*num_true + [{"color":"blue","true_or_med":"medium"}]*num_medium
random.shuffle(test_blue)

test_green = [{"color":"green","true_or_med":"true"}]*num_true + [{"color":"green","true_or_med":"medium"}]*num_medium
random.shuffle(test_green)

# Combine them and pair with angles
# One simple approach: 8 random angles for blue, 8 random angles for green
# We'll just create a combined list, then shuffle.
test_block_blue = []
for i in range(num_each_color):
    test_block_blue.append({
        "angle": angle_list[i % len(angle_list)],
        "color": test_blue[i]["color"],
        "true_or_med": test_blue[i]["true_or_med"]
    })

test_block_green = []
for i in range(num_each_color):
    test_block_green.append({
        "angle": angle_list[i % len(angle_list)],
        "color": test_green[i]["color"],
        "true_or_med": test_green[i]["true_or_med"]
    })

test_conditions_combined = test_block_blue + test_block_green
random.shuffle(test_conditions_combined)

def run_test_trial(angle_bias, color_cue, true_or_med, blue_final, green_final, medium_prop):
    """
    Test-phase trial: if 'true_or_med'=='true', use color's final proportion;
    else use 'medium_prop'.
    """
    if color_cue == "blue":
        fix_color = "blue"
        if true_or_med == "true":
            prop = blue_final
        else:
            prop = medium_prop
    else:
        fix_color = "green"
        if true_or_med == "true":
            prop = green_final
        else:
            prop = medium_prop

    fixation.color = fix_color
    fixation.draw()
    win.flip()
    core.wait(0.5)

    # random positions
    sqX, sqY, dotX, dotY = random_positions()
    square.pos = (sqX, sqY)
    dot.pos    = (dotX, dotY)

    tShape = random.choice(["square","dot"])

    snippet_index = random.randrange(ou_library.shape[0])
    snippet = ou_library[snippet_index]
    mouse = event.Mouse(win=win, visible=False)
    mouse.setPos((0,0))
    distractor_velocity = np.array([0.0, 0.0])
    clock = core.Clock()
    clock.reset()
    last_mx, last_my = 0, 0
    frameN = 0

    while clock.getTime() < 5.0:
        mx, my = mouse.getPos()
        dx = mx - last_mx
        dy = my - last_my
        last_mx, last_my = mx, my

        ou_dx, ou_dy = snippet[frameN % snippet_length]
        mag_mouse   = np.hypot(dx, dy)
        mag_snippet = np.hypot(ou_dx, ou_dy)
        if mag_snippet > 0:
            norm_ou_dx = (ou_dx / mag_snippet) * mag_mouse
            norm_ou_dy = (ou_dy / mag_snippet) * mag_mouse
        else:
            norm_ou_dx, norm_ou_dy = 0, 0

        dx_biased, dy_biased = rotate_vector(dx, dy, angle_bias)
        targ_dx = (1 - prop) * dx_biased + prop * norm_ou_dx
        targ_dy = (1 - prop) * dy_biased + prop * norm_ou_dy

        momentum_coef = 0.8
        current_ou_velocity = np.array([norm_ou_dx, norm_ou_dy])
        distractor_velocity = momentum_coef * distractor_velocity + (1 - momentum_coef) * current_ou_velocity
        dist_dx, dist_dy = distractor_velocity

        if tShape == "square":
            square.pos = confine_position((square.pos[0] + targ_dx, square.pos[1] + targ_dy))
            dot.pos    = confine_position((dot.pos[0] + dist_dx, dot.pos[1] + dist_dy))
        else:
            dot.pos    = confine_position((dot.pos[0] + targ_dx, dot.pos[1] + targ_dy))
            square.pos = confine_position((square.pos[0] + dist_dx, square.pos[1] + dist_dy))

        square.draw()
        dot.draw()
        win.flip()
        frameN += 1

    # 4-point response
    prompt_text = (
        "Which shape did you control, and how confident are you?\n"
        "1 = Square (Guess)\n"
        "2 = Square (Confident)\n"
        "3 = Dot    (Guess)\n"
        "4 = Dot    (Confident)"
    )
    prompt_stim = visual.TextStim(win, text=prompt_text, color="white", height=26)
    prompt_stim.draw()
    win.flip()

    resp_key = event.waitKeys(keyList=["1","2","3","4","escape"])
    if "escape" in resp_key:
        core.quit()

    choice_map = {
        "1": ("square","guess"),
        "2": ("square","confident"),
        "3": ("dot","guess"),
        "4": ("dot","confident")
    }
    chosen_shape, chosen_conf = choice_map[resp_key[0]]
    accuracy = 1 if (chosen_shape == tShape) else 0

    # 0–100 rating
    rating_text = "How much control did you feel (0–100)?"
    rating_stim = visual.TextStim(win, text=rating_text, color="white", height=28)
    rating_stim.draw()
    win.flip()

    rating_string = ""
    finished_rating = False
    while not finished_rating:
        keys = event.waitKeys()
        for k in keys:
            if k in ["return","enter"]:
                finished_rating = True
                break
            elif k == "backspace":
                rating_string = rating_string[:-1]
            elif k.isdigit():
                if len(rating_string) < 3:
                    rating_string += k
            elif k == "escape":
                core.quit()

        display_text = rating_text + f"\nCurrent: {rating_string}"
        rating_stim.text = display_text
        rating_stim.draw()
        win.flip()

    rating_value = float(rating_string) if rating_string else 0
    rating_value = max(0, min(100, rating_value))

    win.flip()
    core.wait(0.3)

    return accuracy, chosen_shape, chosen_conf, rating_value, tShape, prop

test_results = []
for cond in test_conditions_combined:
    angle_bias = cond["angle"]
    color_cue  = cond["color"]
    true_or_med = cond["true_or_med"]

    accuracy, chosen_shape, chosen_conf, rating_value, true_shape, prop_used = run_test_trial(
        angle_bias, color_cue, true_or_med,
        blue_final, green_final, medium_prop
    )

    # Save data
    thisExp.addData("phase", "test")
    thisExp.addData("angle_bias", angle_bias)
    thisExp.addData("color_cue", color_cue)
    thisExp.addData("true_or_med", true_or_med)
    thisExp.addData("true_shape", true_shape)
    thisExp.addData("chosen_shape", chosen_shape)
    thisExp.addData("chosen_conf", chosen_conf)
    thisExp.addData("control_rating_0to100", rating_value)
    thisExp.addData("prop_used", prop_used)
    thisExp.addData("accuracy", accuracy)
    thisExp.nextEntry()

#############################################
# 7. Save & Close
#############################################

csv_path = os.path.join(data_path, filename + ".csv")
thisExp.saveAsWideText(csv_path)
print(f"Data saved to: {csv_path}")
win.close()
core.quit()
