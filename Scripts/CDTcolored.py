#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Control Detection Task with:
 - Two Staircases (Blue ~60% / Green ~90%)
 - Combined control+confidence (4-point) + 0–100 rating
 - Training & Test phases
 - Movement-generation approach from the 'Demo' snippet (OU snippet + momentum)
 - Data saving on premature exit
"""

import os, random, numpy as np
from psychopy import visual, core, data, event, gui

#############################################
# 1. Experiment Setup
#############################################

expName = "ControlDetection_TwoStaircases_WithDemoMovement"
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
    """Random non-overlapping positions for square & dot within radius=250."""
    while True:
        x1 = random.uniform(-250, 250)
        y1 = random.uniform(-250, 250)
        if np.hypot(x1, y1) <= 250:
            x2 = random.uniform(-250, 250)
            y2 = random.uniform(-250, 250)
            if np.hypot(x2, y2) <= 250:
                if np.hypot(x1 - x2, y1 - y2) >= 200:
                    return (x1, y1, x2, y2)

# Define stimuli
square   = visual.Rect(win, width=40, height=40, fillColor="black", lineColor="black")
dot      = visual.Circle(win, radius=20, fillColor="black", lineColor="black")
fixation = visual.TextStim(win, text="+", color="white", height=30)
instruction_text = visual.TextStim(win, text="", color="white", height=28, wrapWidth=1000)

def generate_OU_snippet(n_frames=300, theta=0.2, sigma=2.0, dt=1.0,
                        linear_bias=(0,0), angular_bias=0):
    """
    Generate an OU snippet (same as the Demo code).
    """
    v = np.zeros((n_frames, 2))
    for t in range(1, n_frames):
        v[t,0] = v[t-1,0] - theta * v[t-1,0] * dt + sigma * np.random.randn() * np.sqrt(dt) + linear_bias[0]*dt
        v[t,1] = v[t-1,1] - theta * v[t-1,1] * dt + sigma * np.random.randn() * np.sqrt(dt) + linear_bias[1]*dt
        speed = np.hypot(v[t,0], v[t,1])
        angle = np.arctan2(v[t,1], v[t,0]) + angular_bias * dt
        v[t,0] = speed * np.cos(angle)
        v[t,1] = speed * np.sin(angle)
    return v

#############################################
# 3. Free Movement Demo (to estimate bias)
#############################################

instruction_text.text = (
    "FREE-MOVEMENT DEMO (30s)\n\n"
    "Move your mouse for ~30s. One shape is partly under your control.\n"
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
tShape = random.choice(["square","dot"])

pre_trial_n_frames = int(pre_trial_duration * 60)
dummy_snippet = generate_OU_snippet(n_frames=pre_trial_n_frames, theta=0.2, sigma=2.0, dt=1.0)

distractor_velocity = np.array([0.0, 0.0])
momentum_coef = 0.8
prop_demo = 0.6

mouse = event.Mouse(win=win, visible=True)
mouse.setPos((0,0))
last_mx, last_my = mouse.getPos()
pre_movements = []
pre_clock = core.Clock()
pre_clock.reset()
frameN = 0

while pre_clock.getTime() < pre_trial_duration:
    mx, my = mouse.getPos()
    dx = mx - last_mx
    dy = my - last_my
    pre_movements.append((dx, dy))
    last_mx, last_my = mx, my

    ou_dx, ou_dy = dummy_snippet[frameN % pre_trial_n_frames]
    mag_mouse = np.hypot(dx, dy)
    mag_snippet = np.hypot(ou_dx, ou_dy)
    if mag_snippet > 0:
        norm_ou_dx = (ou_dx/mag_snippet)*mag_mouse
        norm_ou_dy = (ou_dy/mag_snippet)*mag_mouse
    else:
        norm_ou_dx, norm_ou_dy = 0, 0

    targ_dx = (1 - prop_demo)*dx + prop_demo*norm_ou_dx
    targ_dy = (1 - prop_demo)*dy + prop_demo*norm_ou_dy

    current_ou_velocity = np.array([norm_ou_dx, norm_ou_dy])
    distractor_velocity = momentum_coef*distractor_velocity + (1 - momentum_coef)*current_ou_velocity
    dist_dx, dist_dy = distractor_velocity

    if tShape == "square":
        square.pos = confine_position((square.pos[0]+targ_dx, square.pos[1]+targ_dy), 250)
        dot.pos    = confine_position((dot.pos[0]+dist_dx, dot.pos[1]+dist_dy), 250)
    else:
        dot.pos    = confine_position((dot.pos[0]+targ_dx, dot.pos[1]+targ_dy), 250)
        square.pos = confine_position((square.pos[0]+dist_dx, square.pos[1]+dist_dy), 250)

    square.draw()
    dot.draw()
    win.flip()
    frameN += 1

choice_text = visual.TextStim(win, text="Which shape did you control more?\n(S) Square   (D) Dot", color="white", height=30)
choice_text.draw()
win.flip()
key = event.waitKeys(keyList=["s", "d", "escape"])
if "escape" in key:
    core.quit()
pre_choice = "square" if key[0]=="s" else "dot"

pre_movements = np.array(pre_movements)
avg_linear_bias = pre_movements.mean(axis=0)
angles = np.arctan2(pre_movements[:,1], pre_movements[:,0])
angular_changes = np.diff(angles)
angular_changes = (angular_changes + np.pi) % (2*np.pi) - np.pi
avg_angular_bias = angular_changes.mean() if angular_changes.size>0 else 0

#############################################
# 4. Generate OU Library for Main Task (with bias)
#############################################

snippet_length = 300
num_snippets = 2000
ou_library = []
for _ in range(num_snippets):
    s = generate_OU_snippet(n_frames=snippet_length, theta=0.2, sigma=2.0, dt=1.0,
                            linear_bias=avg_linear_bias, angular_bias=avg_angular_bias)
    ou_library.append(s)
ou_library = np.array(ou_library)

#############################################
# 5. Two Staircases (Practice Phase): Blue ~60%, Green ~90%
#############################################

blue_prop = 0.50
green_prop = 0.70
blue_correct = 0
blue_total = 0
green_correct = 0
green_total = 0
TARGET_BLUE = 0.60
TARGET_GREEN = 0.90
STEP_SIZE = 0.05

angles_practice = [0]*5 + [90]*5
random.shuffle(angles_practice)
colors_practice = ["blue"]*5 + ["green"]*5
random.shuffle(colors_practice)
practice_conditions = list(zip(angles_practice, colors_practice))
random.shuffle(practice_conditions)

instruction_text.text = (
    "PRACTICE PHASE (10 trials):\n"
    " - Color cues: blue (~60%) or green (~90%).\n"
    " - Each trial also has angle 0° or 90°.\n\n"
    "Press any key to continue."
)
instruction_text.draw()
win.flip()
event.waitKeys()

def run_practice_trial(angle_bias, color_cue, blue_prop, green_prop):
    """
    Single practice trial using the movement approach from the Demo snippet.
    Returns: (accuracy, chosen_shape, chosen_conf, rating_value, true_shape).
    """
    if color_cue == "blue":
        fix_color = "blue"
        prop_trial = blue_prop
    else:
        fix_color = "green"
        prop_trial = green_prop

    fixation.color = fix_color
    square.fillColor = fix_color
    square.lineColor = fix_color
    dot.fillColor = fix_color
    dot.lineColor = fix_color

    fixation.draw()
    win.flip()
    core.wait(0.5)

    sqX, sqY, dotX, dotY = random_positions()
    square.pos = (sqX, sqY)
    dot.pos = (dotX, dotY)
    tShape = random.choice(["square", "dot"])

    snippet_index = random.randrange(ou_library.shape[0])
    snippet = ou_library[snippet_index]
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

        ou_dx, ou_dy = snippet[frameN % snippet_length]
        dx_biased, dy_biased = rotate_vector(dx, dy, angle_bias)
        mag_mouse = np.hypot(dx_biased, dy_biased)
        mag_snippet = np.hypot(ou_dx, ou_dy)
        if mag_snippet > 0:
            norm_ou_dx = (ou_dx/mag_snippet)*mag_mouse
            norm_ou_dy = (ou_dy/mag_snippet)*mag_mouse
        else:
            norm_ou_dx, norm_ou_dy = 0, 0

        targ_dx = (1 - prop_trial)*dx_biased + prop_trial*norm_ou_dx
        targ_dy = (1 - prop_trial)*dy_biased + prop_trial*norm_ou_dy

        current_ou_velocity = np.array([norm_ou_dx, norm_ou_dy])
        distractor_velocity = momentum_coef*distractor_velocity + (1 - momentum_coef)*current_ou_velocity
        dist_dx, dist_dy = distractor_velocity

        if tShape == "square":
            square.pos = confine_position((square.pos[0]+targ_dx, square.pos[1]+targ_dy))
            dot.pos = confine_position((dot.pos[0]+dist_dx, dot.pos[1]+dist_dy))
        else:
            dot.pos = confine_position((dot.pos[0]+targ_dx, dot.pos[1]+targ_dy))
            square.pos = confine_position((square.pos[0]+dist_dx, square.pos[1]+dist_dy))

        square.draw()
        dot.draw()
        win.flip()
        frameN += 1

    resp_text = (
        "Which shape did you control, and how confident are you?\n"
        "1 = Square (Guess)\n"
        "2 = Square (Confident)\n"
        "3 = Dot    (Guess)\n"
        "4 = Dot    (Confident)"
    )
    resp_stim = visual.TextStim(win, text=resp_text, color="white", height=26)
    resp_stim.draw()
    win.flip()
    key_resp = event.waitKeys(keyList=["1","2","3","4","escape"])
    if "escape" in key_resp:
        core.quit()
    choice_map = {"1":("square","guess"),
                  "2":("square","confident"),
                  "3":("dot","guess"),
                  "4":("dot","confident")}
    chosen_shape, chosen_conf = choice_map[key_resp[0]]
    accuracy = 1 if (chosen_shape == tShape) else 0

    rating_text = "How much control did you feel (0–100)?"
    rating_stim = visual.TextStim(win, text=rating_text, color="white", height=28)
    rating_stim.draw()
    win.flip()
    rating_string = ""
    done_rating = False
    while not done_rating:
        keys = event.waitKeys()
        for k in keys:
            if k in ["return","enter"]:
                done_rating = True
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
    core.wait(0.3)
    return accuracy, chosen_shape, chosen_conf, rating_value, tShape

#############################################
# 6. TEST PHASE (True vs. Medium; 0° vs. 90°)
#############################################

# Prepare test conditions: For each color, 8 trials (half true, half medium)
num_each_color = 8
num_true = 4
num_med = 4
angles_0 = [0]*(num_each_color//2)
angles_90 = [90]*(num_each_color//2)
angle_list = angles_0 + angles_90
random.shuffle(angle_list)

test_blue = [{"color":"blue","true_or_med":"true"}]*num_true + [{"color":"blue","true_or_med":"medium"}]*num_med
random.shuffle(test_blue)
test_green = [{"color":"green","true_or_med":"true"}]*num_true + [{"color":"green","true_or_med":"medium"}]*num_med
random.shuffle(test_green)

test_block_blue = []
for i in range(num_each_color):
    test_block_blue.append({"angle": angle_list[i % len(angle_list)],
                            "color": test_blue[i]["color"],
                            "true_or_med": test_blue[i]["true_or_med"]})
test_block_green = []
for i in range(num_each_color):
    test_block_green.append({"angle": angle_list[i % len(angle_list)],
                             "color": test_green[i]["color"],
                             "true_or_med": test_green[i]["true_or_med"]})
test_conditions_combined = test_block_blue + test_block_green
random.shuffle(test_conditions_combined)

def run_test_trial(angle_bias, color_cue, true_or_med, blue_final, green_final, medium_prop):
    if color_cue=="blue":
        fix_color = "blue"
        prop_trial = blue_final if true_or_med=="true" else medium_prop
    else:
        fix_color = "green"
        prop_trial = green_final if true_or_med=="true" else medium_prop

    fixation.color = fix_color
    square.fillColor = fix_color
    square.lineColor = fix_color
    dot.fillColor = fix_color
    dot.lineColor = fix_color

    fixation.draw()
    win.flip()
    core.wait(0.5)

    sqX, sqY, dotX, dotY = random_positions()
    square.pos = (sqX, sqY)
    dot.pos = (dotX, dotY)
    tShape = random.choice(["square", "dot"])

    snippet_index = random.randrange(ou_library.shape[0])
    snippet = ou_library[snippet_index]
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
        ou_dx, ou_dy = snippet[frameN % snippet_length]
        dx_biased, dy_biased = rotate_vector(dx, dy, angle_bias)
        mag_mouse = np.hypot(dx_biased, dy_biased)
        mag_snippet = np.hypot(ou_dx, ou_dy)
        if mag_snippet > 0:
            norm_ou_dx = (ou_dx/mag_snippet)*mag_mouse
            norm_ou_dy = (ou_dy/mag_snippet)*mag_mouse
        else:
            norm_ou_dx, norm_ou_dy = 0, 0

        targ_dx = (1 - prop_trial)*dx_biased + prop_trial*norm_ou_dx
        targ_dy = (1 - prop_trial)*dy_biased + prop_trial*norm_ou_dy

        current_ou_velocity = np.array([norm_ou_dx, norm_ou_dy])
        distractor_velocity = momentum_coef*distractor_velocity + (1 - momentum_coef)*current_ou_velocity
        dist_dx, dist_dy = distractor_velocity

        if tShape == "square":
            square.pos = confine_position((square.pos[0]+targ_dx, square.pos[1]+targ_dy))
            dot.pos = confine_position((dot.pos[0]+dist_dx, dot.pos[1]+dist_dy))
        else:
            dot.pos = confine_position((dot.pos[0]+targ_dx, dot.pos[1]+targ_dy))
            square.pos = confine_position((square.pos[0]+dist_dx, square.pos[1]+dist_dy))

        square.draw()
        dot.draw()
        win.flip()
        frameN += 1

    resp_text = (
        "Which shape did you control, and how confident are you?\n"
        "1 = Square (Guess)\n"
        "2 = Square (Confident)\n"
        "3 = Dot    (Guess)\n"
        "4 = Dot    (Confident)"
    )
    resp_stim = visual.TextStim(win, text=resp_text, color="white", height=26)
    resp_stim.draw()
    win.flip()
    key_resp = event.waitKeys(keyList=["1","2","3","4","escape"])
    if "escape" in key_resp:
        core.quit()
    choice_map = {"1":("square","guess"),
                  "2":("square","confident"),
                  "3":("dot","guess"),
                  "4":("dot","confident")}
    chosen_shape, chosen_conf = choice_map[key_resp[0]]
    accuracy = 1 if (chosen_shape == tShape) else 0

    rating_text = "How much control did you feel (0–100)?"
    rating_stim = visual.TextStim(win, text=rating_text, color="white", height=28)
    rating_stim.draw()
    win.flip()
    rating_string = ""
    done_rating = False
    while not done_rating:
        keys = event.waitKeys()
        for k in keys:
            if k in ["return","enter"]:
                done_rating = True
                break
            elif k == "backspace":
                rating_string = rating_string[:-1]
            elif k.isdigit():
                if len(rating_string) < 3:
                    rating_string += k
            elif k=="escape":
                core.quit()
        display_text = rating_text + f"\nCurrent: {rating_string}"
        rating_stim.text = display_text
        rating_stim.draw()
        win.flip()
    rating_value = float(rating_string) if rating_string else 0
    rating_value = max(0, min(100, rating_value))
    core.wait(0.3)
    return accuracy, chosen_shape, chosen_conf, rating_value, tShape, prop_trial

#############################################
# Main Experiment: Practice & Test Phases
#############################################

# Wrap the main experiment code in try/except/finally so that data are saved on premature exit.
try:
    # --- Practice Phase ---
    for angle_bias, color_cue in practice_conditions:
        acc, chosen_shape, chosen_conf, rate_val, true_shape = run_practice_trial(angle_bias, color_cue, blue_prop, green_prop)
        if color_cue=="blue":
            blue_total += 1
            blue_correct += acc
            cur_acc = blue_correct/float(blue_total)
            if cur_acc > TARGET_BLUE:
                blue_prop = max(0.0, blue_prop - STEP_SIZE)
            else:
                blue_prop = min(1.0, blue_prop + STEP_SIZE)
        else:
            green_total += 1
            green_correct += acc
            cur_acc = green_correct/float(green_total)
            if cur_acc > TARGET_GREEN:
                green_prop = max(0.0, green_prop - STEP_SIZE)
            else:
                green_prop = min(1.0, green_prop + STEP_SIZE)
        thisExp.addData("phase", "practice")
        thisExp.addData("angle_bias", angle_bias)
        thisExp.addData("color_cue", color_cue)
        thisExp.addData("true_shape", true_shape)
        thisExp.addData("chosen_shape", chosen_shape)
        thisExp.addData("chosen_conf", chosen_conf)
        thisExp.addData("control_rating_0to100", rate_val)
        thisExp.addData("accuracy", acc)
        thisExp.addData("prop_used", blue_prop if color_cue=="blue" else green_prop)
        thisExp.addData("running_accuracy", cur_acc)
        thisExp.nextEntry()

    blue_final = blue_prop
    green_final = green_prop
    medium_prop = (blue_final + green_final) / 2.0
    print(f"Blue final: {blue_final}, Green final: {green_final}, Medium: {medium_prop}")

    # --- Test Phase ---
    instruction_text.text = (
        "TEST PHASE:\n"
        "Each color cue is sometimes 'true' (its final staircase value) and\n"
        "sometimes 'medium' (the average of blue and green).\n\n"
        "Press any key to continue."
    )
    instruction_text.draw()
    win.flip()
    event.waitKeys()

    # Prepare test conditions as before
    test_conditions = []
    num_each_color = 8
    num_true = 4
    num_med = 4
    angles_0 = [0]*(num_each_color//2)
    angles_90 = [90]*(num_each_color//2)
    angle_list = angles_0 + angles_90
    random.shuffle(angle_list)
    test_blue = [{"color":"blue", "true_or_med":"true"}]*num_true + [{"color":"blue", "true_or_med":"medium"}]*num_med
    random.shuffle(test_blue)
    test_green = [{"color":"green", "true_or_med":"true"}]*num_true + [{"color":"green", "true_or_med":"medium"}]*num_med
    random.shuffle(test_green)
    test_block_blue = []
    for i in range(num_each_color):
        test_block_blue.append({"angle": angle_list[i % len(angle_list)],
                                "color": test_blue[i]["color"],
                                "true_or_med": test_blue[i]["true_or_med"]})
    test_block_green = []
    for i in range(num_each_color):
        test_block_green.append({"angle": angle_list[i % len(angle_list)],
                                 "color": test_green[i]["color"],
                                 "true_or_med": test_green[i]["true_or_med"]})
    test_conditions_combined = test_block_blue + test_block_green
    random.shuffle(test_conditions_combined)

    for cond in test_conditions_combined:
        angle_bias = cond["angle"]
        color_cue = cond["color"]
        true_or_med = cond["true_or_med"]
        acc, chosen_shape, chosen_conf, rate_val, true_shape, prop_used = run_test_trial(angle_bias, color_cue, true_or_med, blue_final, green_final, medium_prop)
        thisExp.addData("phase", "test")
        thisExp.addData("angle_bias", angle_bias)
        thisExp.addData("color_cue", color_cue)
        thisExp.addData("true_or_med", true_or_med)
        thisExp.addData("true_shape", true_shape)
        thisExp.addData("chosen_shape", chosen_shape)
        thisExp.addData("chosen_conf", chosen_conf)
        thisExp.addData("control_rating_0to100", rate_val)
        thisExp.addData("prop_used", prop_used)
        thisExp.addData("accuracy", acc)
        thisExp.nextEntry()

except KeyboardInterrupt:
    print("Experiment terminated prematurely. Saving data...")

finally:
    csv_path = os.path.join(data_path, filename + ".csv")
    thisExp.saveAsWideText(csv_path)
    print(f"Data saved to: {csv_path}")
    win.close()
    core.quit()
