#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Free Movement Collection Task

Participants are instructed to move the mouse freely (i.e. “draw”) on the screen for 5 seconds.
After each trial, they rate the naturalness of their movement on a 1-5 scale.
All mouse positions (the trajectory) are saved for later analysis.
This experiment can yield a library of naturalistic movements for use in later experiments.
"""

from psychopy import visual, core, data, event, gui, logging
import numpy as np
import os, random

# ---------------------------
# Experiment Setup
# ---------------------------
expName = "FreeMovementCollection"
expInfo = {"participant": "", "session": "001"}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if not dlg.OK:
    core.quit()

# Data file naming
filename = f"{expName}_P{expInfo['participant']}_S{expInfo['session']}_{data.getDateStr()}"
if not os.path.isdir("data"):
    os.makedirs("data")

thisExp = data.ExperimentHandler(name=expName,
                                 extraInfo=expInfo,
                                 dataFileName=os.path.join("data", filename))

# Setup logging
logging.console.setLevel(logging.WARNING)
logFile = logging.LogFile(os.path.join("data", filename + ".log"), level=logging.EXP)

# Create window
win = visual.Window(size=[800,600], color=[0.5,0.5,0.5], units="pix")

# ---------------------------
# Stimuli & Instructions
# ---------------------------
instruction_text = visual.TextStim(win, text=(
    "Free Movement Task\n\n"
    "In each trial, move the mouse freely for 5 seconds. Try to produce natural, smooth, and varied movements.\n"
    "After each trial, rate how natural your movement felt on a scale from 1 (not natural) to 5 (very natural).\n\n"
    "Press any key to begin."
), height=20, color="white", wrapWidth=700)

rating_prompt = visual.TextStim(win, text="Rate the naturalness of your movement (1-5):", 
                                height=20, color="white", pos=(0,50))
rating_display = visual.TextStim(win, text="", height=30, color="yellow", pos=(0,0))
trial_info_text = visual.TextStim(win, text="", height=20, color="white", pos=(0,-250))

# ---------------------------
# Task Parameters
# ---------------------------
trial_duration = 5.0  # seconds per trial
n_trials = 50  # adjust this to collect many trials; you can also increase the number per participant

# ---------------------------
# Instructions
# ---------------------------
instruction_text.draw()
win.flip()
event.waitKeys()

# ---------------------------
# Trial Loop
# ---------------------------
all_trials_data = []  # list to store data from each trial

for trial in range(n_trials):
    # Start trial: clear mouse trajectory record
    mouse = event.Mouse(win=win, visible=True)
    mouse.setPos((0,0))
    traj = []  # to record positions
    trial_clock = core.Clock()
    
    # Display a blank screen with a fixation cross (optional)
    fixation = visual.TextStim(win, text="+", height=40, color="white")
    fixation.draw()
    win.flip()
    core.wait(0.5)
    
    # Start movement trial
    trial_clock.reset()
    while trial_clock.getTime() < trial_duration:
        pos = mouse.getPos()
        traj.append(pos)
        # Optionally draw a dot at current position (or leave the screen blank)
        current_dot = visual.Circle(win, radius=5, fillColor="red", pos=pos)
        current_dot.draw()
        win.flip()
    
    # After the trial, prompt for a rating
    rating = None
    rating_str = ""
    rating_prompt.draw()
    rating_display.setText(rating_str)
    win.flip()
    
    # Use a simple text input loop for rating:
    while rating is None:
        keys = event.waitKeys()
        for k in keys:
            if k in ["1","2","3","4","5"]:
                rating_str += k
                rating_display.setText(rating_str)
                rating_prompt.draw()
                rating_display.draw()
                win.flip()
            elif k == "return" and rating_str != "":
                rating = int(rating_str)
    
    # Save trial data: record the entire trajectory, rating, and trial info
    trial_data = {
        "trial": trial+1,
        "trajectory": traj,  # list of (x,y) positions sampled every frame
        "rating": rating,
        "trial_duration": trial_duration
    }
    all_trials_data.append(trial_data)
    
    # Show a message for the next trial
    trial_info_text.setText(f"Trial {trial+1}/{n_trials} complete. Press any key for next trial.")
    trial_info_text.draw()
    win.flip()
    event.waitKeys()
    
# End of experiment
instruction_text.setText("Thank you for participating!")
instruction_text.draw()
win.flip()
core.wait(3.0)

# Save the data to file. You might want to use pickle or numpy.save if saving large arrays.
import pickle
with open(os.path.join("data", filename + ".pkl"), "wb") as f:
    pickle.dump(all_trials_data, f)

thisExp.addData("all_trials_data", all_trials_data)
thisExp.saveAsWideText(os.path.join("data", filename + ".csv"))
thisExp.saveAsPickle(os.path.join("data", filename))
win.close()
core.quit()
