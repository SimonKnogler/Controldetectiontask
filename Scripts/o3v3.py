#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wp1_control_detection.py

WORK PACKAGE 1   –   Control-detection task with confidence & agency
Author:  …
Date  :  2025-05-05

--------------------------------------------------------------------
Key features requested by the user
--------------------------------------------------------------------
✓  Two expectation conditions (High, Low) signalled by colours
✓  Two processing modes     (0° vs. 90° angular bias)
✓  Weighted-Up-Down staircases that aim at 80 % (High) and 60 % (Low)    –  see Kaernbach 1991 BF03214307 (1).pdf](file-service://file-C6Ewfa1Kc5cNNiRpCagPxG)
✓  5-s trial length, 5 practice trials per cell (total = 20)
✓  Medium-strength “test-phase” proportion = average of the two staircases
✓  Smooth analytic random motion  (no prerecorded traces needed)
✓  Data columns: angularBias, expectation, shapeChosen, shapeCorrect,
                 accuracy, confidence, selfMotionProp, RT
✓  Graceful early exit with Esc that still writes the CSV
"""

# -------------------------------------------------------------------
# 1.  Imports
# -------------------------------------------------------------------
from psychopy import visual, event, core, gui, data, logging  # PsychoPy core
import numpy as np                                           # maths
import pandas as pd                                          # convenient CSV writing
import random                                                # misc RNG
import os, sys, datetime                                     # housekeeping

# -------------------------------------------------------------------
# 2.  Experiment-wide constants
# -------------------------------------------------------------------

# --- visual parameters ------------------------------------------------
WIN_SIZE          = [1280, 720]      # window resolution (px)
BG_COLOR          = [-1, -1, -1]     # black (PsychoPy uses −1..+1)
FIX_DUR           = 0.5              # fixation duration in s
TRIAL_DUR         = 5.0              # movement period in s

# --- stimuli ----------------------------------------------------------
SHAPE_SIZE        = 0.05             # height in norm units (~5 % of screen)
CONTROL_COLOR     = [1, 1, 1]        # white shapes
EXPECT_COLORS     = {'high': 'blue', # colour of fixation + shapes
                     'low' : 'green',
                     'medium':'purple'}

# --- staircase parameters --------------------------------------------
STEP_BASE         = 0.05             # basic step in proportion units (0…1)
STAIR_HIGH        = dict(pTarget=.80,  stepUp=STEP_BASE*.25, stepDown=STEP_BASE)   # Sup/Sdown = 0.25
STAIR_LOW         = dict(pTarget=.60,  stepUp=STEP_BASE*.67, stepDown=STEP_BASE)   # Sup/Sdown = 0.67
#  Note: ratios follow Sup/Sdown = (1-p)/p  (Kaernbach 1991) BF03214307 (1).pdf](file-service://file-C6Ewfa1Kc5cNNiRpCagPxG)

# --- experiment design : 5 trials × 2 (expect) × 2 (bias) --------------
TRIALS_PER_CELL   = 5
CONDITIONS        = [(expect, bias)
                     for expect in ('low', 'high')
                     for bias   in (0, 90)]                 # 4 cells

# --- response mapping -------------------------------------------------
KEY_SHAPES        = {'square': 'left', 'circle': 'right'}   # keys for decision
KEY_CONF          = ['1','2','3','4']                       # 1=high-conf … 4=guess
EXIT_KEY          = 'escape'

# -------------------------------------------------------------------
# 3.  Weighted-Up-Down staircase class
# -------------------------------------------------------------------
class WeightedUD:
    """ Minimal weighted up-down staircase (Kaernbach 1991). """

    def __init__(self, startProp, stepUp, stepDown, pTarget):
        self.prop        = startProp
        self.stepUp      = stepUp
        self.stepDown    = stepDown
        self.pTarget     = pTarget
        self.history     = []        # store tuples (correct?, prop)

    def update(self, correct):
        """Update proportion after a response and return new value."""
        if correct:                   # task too easy → harder
            self.prop = max(0.0,
                             self.prop - self.stepDown)
        else:                         # task too hard → easier
            self.prop = min(1.0,
                             self.prop + self.stepUp)
        self.history.append((correct, self.prop))
        return self.prop

# -------------------------------------------------------------------
# 4.  Smooth random movement generator (Ornstein-Uhlenbeck)
# -------------------------------------------------------------------
def smooth_random(nFrames, dt, tau=0.15, sigma=120):
    """
    Generate a smooth 2-D random walk.
    Ornstein-Uhlenbeck solves jitteriness by autocorrelation.
      nFrames : number of points
      dt      : frame time (≈ 1/60 s)
      tau     : mean-reversion time [s]  (smaller → more wiggly)
      sigma   : spatial scale [px/s]
    Returns an (nFrames, 2) array of x,y displacements.
    """
    vel = np.zeros((nFrames, 2))
    std = sigma * np.sqrt((1-np.exp(-2*dt/tau)))
    for i in range(1, nFrames):
        vel[i] = vel[i-1]*np.exp(-dt/tau) + np.random.randn(2)*std
    return vel * dt                           # integrate velocity → displacement

# -------------------------------------------------------------------
# 5.  Initialise PsychoPy window and stimuli
# -------------------------------------------------------------------
win          = visual.Window(WIN_SIZE, color=BG_COLOR, units='norm')
fixCross     = visual.TextStim(win, text='+', color='white', height=0.05)
squareStim   = visual.Rect (win, width=SHAPE_SIZE, height=SHAPE_SIZE,
                            fillColor=CONTROL_COLOR, lineColor=CONTROL_COLOR)
circleStim   = visual.Circle(win, radius=SHAPE_SIZE/2,
                            fillColor=CONTROL_COLOR, lineColor=CONTROL_COLOR)

# -------------------------------------------------------------------
# 6.  Participant & file handling
# -------------------------------------------------------------------
dlg  = gui.Dlg(title="WP1 Control-Detection")
dlg.addText("Participant information")
dlg.addField("ID :")
dlg.show()
if dlg.OK:
    subjID = dlg.data[0]
else:
    core.quit()

filename   = f"data/{subjID}_{datetime.date.today()}.csv"
os.makedirs('data', exist_ok=True)

# container for per-trial data
rows = []

# -------------------------------------------------------------------
# 7.  Create two staircases (practice phase)
# -------------------------------------------------------------------
stairStart      = 0.65             # initial proportion of self-motion
staircases      = {'high': WeightedUD(stairStart, **STAIR_HIGH),
                   'low' : WeightedUD(stairStart, **STAIR_LOW)}

# -------------------------------------------------------------------
# 8.  Prepare trial list  (20 practice trials, random order)
# -------------------------------------------------------------------
trialList = []
for expCond, bias in CONDITIONS:
    for t in range(TRIALS_PER_CELL):
        trialList.append({'expect': expCond,
                          'bias'  : bias})
random.shuffle(trialList)          # randomise order

# -------------------------------------------------------------------
# 9.  Main experiment loop (practice)
# -------------------------------------------------------------------
globalClock = core.Clock()
for trialN, trial in enumerate(trialList, start=1):

    # Abort check -----------------------------------------------------
    if event.getKeys([EXIT_KEY]):
        break                          # exit the for-loop and save data

    # Unpack condition -----------------------------------------------
    expectStr   = trial['expect']      # 'high' | 'low'
    biasAngle   = trial['bias']        # 0 | 90

    # Get current staircase proportion -------------------------------
    propSelf    = staircases[expectStr].prop

    # Colour stimuli/fixation accordingly ----------------------------
    fixCross.color       = EXPECT_COLORS[expectStr]
    squareStim.fillColor = EXPECT_COLORS[expectStr]
    circleStim.fillColor = EXPECT_COLORS[expectStr]

    # Show fixation ---------------------------------------------------
    fixCross.draw()
    win.flip()
    core.wait(FIX_DUR)

    # Reset mouse & timers -------------------------------------------
    mouse           = event.Mouse(visible=False, win=win)
    trialClock      = core.Clock()
    rtClock         = core.Clock()
    correctChosen   = None             # will be True/False later
    chosenShape     = None             # 'square' | 'circle'

    # Pre-allocate positions -----------------------------------------
    squarePos   = np.array([ -.2,  0]) # left – norm units
    circlePos   = np.array([ +.2,  0]) # right
    squareStim.pos = squarePos
    circleStim.pos = circlePos

    # Prepare smooth random motion for distractor --------------------
    nFrames = int(TRIAL_DUR * 60)      # assume 60 Hz
    dt      = 1/60.0
    randWalk = smooth_random(nFrames, dt)   # px → will convert

    # Main frame loop (= movement period) ----------------------------
    lastMousePos = np.array(mouse.getPos())
    frameIdx     = 0
    while trialClock.getTime() < TRIAL_DUR:

        # --- handle Esc inside trial --------------------------------
        if event.getKeys([EXIT_KEY]):
            break

        # --- obtain mouse delta -------------------------------------
        currentPos = np.array(mouse.getPos())
        delta      = currentPos - lastMousePos
        lastMousePos = currentPos

        # --- rotate delta for bias of *target* shape ----------------
        if biasAngle == 90:
            rotMat = np.array([[0,-1],[1,0]])   # 90° CCW
            deltaT = rotMat @ delta
        else:
            deltaT = delta.copy()

        # --- get distractor displacement for this frame -------------
        randDxDy   = randWalk[frameIdx] / (WIN_SIZE[0]/2)  # px→norm
        frameIdx  += 1

        # Decide which shape is target for this trial ----------------
        if trialN % 2 == 0:            # simple alternation (50/50)
            targetStim     = squareStim
            distractorStim = circleStim
            targetPos      = squarePos
            distractorPos  = circlePos
            targetName     = 'square'
            distractorName = 'circle'
        else:
            targetStim     = circleStim
            distractorStim = squareStim
            targetPos      = circlePos
            distractorPos  = squarePos
            targetName     = 'circle'
            distractorName = 'square'

        # --- compute new positions ----------------------------------
        targetPos      = targetPos      + propSelf * deltaT + (1-propSelf)*randDxDy
        distractorPos  = distractorPos  + randDxDy

        # save back for next frame
        if targetName == 'square':
            squarePos   = targetPos
            circlePos   = distractorPos
        else:
            circlePos   = targetPos
            squarePos   = distractorPos
        squareStim.pos = squarePos
        circleStim.pos = circlePos

        # --- draw & flip -------------------------------------------
        squareStim.draw()
        circleStim.draw()
        win.flip()

    # ----------------------------------------------------------------
    # 10.  Decision phase  (shape choice)
    # ----------------------------------------------------------------
    decisionMsg = visual.TextStim(win,
                  text="Which object did you control?\n\n"
                       "Square  = [left]    Circle = [right]",
                  color='white', height=0.06)
    decisionMsg.draw(); win.flip()
    rtClock.reset()
    keys = event.waitKeys(keyList=list(KEY_SHAPES.values()) + [EXIT_KEY],
                          timeStamped=rtClock)
    if keys[0][0] == EXIT_KEY:    # user abort
        break
    chosenShape = 'square' if keys[0][0]=='left' else 'circle'
    RT          = keys[0][1]

    # Accuracy -------------------------------------------------------
    correctChosen = (chosenShape == targetName)
    staircases[expectStr].update(correctChosen)      # staircase update

    # ----------------------------------------------------------------
    # 11.  Confidence rating (4-point)
    # ----------------------------------------------------------------
    confMsg = visual.TextStim(win,
              text="How confident are you?\n"
                   "1 = very sure … 4 = guess",
              color='white', height=0.06)
    confMsg.draw(); win.flip()
    confKeys = event.waitKeys(keyList=KEY_CONF + [EXIT_KEY],
                              timeStamped=rtClock)
    if confKeys[0][0] == EXIT_KEY:
        break
    confLevel = int(confKeys[0][0])   # 1..4

    # ----------------------------------------------------------------
    # 12.  Save row to list
    # ----------------------------------------------------------------
    rows.append(dict(
        subjID          = subjID,
        trialN          = trialN,
        expect          = expectStr,
        angularBias     = biasAngle,
        shapeChosen     = chosenShape,
        shapeCorrect    = targetName,
        accuracy        = int(correctChosen),
        confidence      = confLevel,
        selfMotionProp  = propSelf,
        RT              = RT
    ))

# -------------------------------------------------------------------
# 13.  Compute medium proportion (simple mean) for test phase
#       –  placeholder; you can insert an actual second phase later.
# -------------------------------------------------------------------
propHigh = staircases['high'].prop
propLow  = staircases['low' ].prop
propMed  = (propHigh + propLow) / 2.0

print(f"\nPractice finished.  High={propHigh:.3f}, Low={propLow:.3f}, "
      f"→ Medium={propMed:.3f}")

# -------------------------------------------------------------------
# 14.  Write CSV (even if aborted)
# -------------------------------------------------------------------
df = pd.DataFrame(rows)
df.to_csv(filename, index=False)
print(f"Saved data to {filename}")

# -------------------------------------------------------------------
# 15.  Tidy up
# -------------------------------------------------------------------
win.close()
core.quit()