#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_motion_library_balanced_3s.py  – v17.1
───────────────────────────────────────────
• Records exactly 5 min (=18 000 *moving* frames) at 60 Hz.
• Each 3-s snippet starts only after the donor moves the mouse.
• Global timer pauses while shapes are frozen for the click.
• Cursor visible at all times.
• Shapes always start each snippet centrally (± offset), left/right counterbalanced.
• Saves
    donor_<ID>_<TS>_snips.npy   (100,180,2) float16
    donor_<ID>_<TS>_<feat.npy>  (100,6)     float32
  to motion_donors/ next to this script.
"""

import math, pathlib, datetime, numpy as np
from psychopy import visual, event, core, gui

# ───────────────────────────────────────────────────────
# 1  Constants
# ───────────────────────────────────────────────────────
FPS         = 60
DT          = 1.0 / FPS
DUR_FRAMES  = 300 * FPS           # 18 000 moving frames = 5 min
SNIP_LEN    = 180                 # 3-second snippet
PROP        = 0.50                # self-motion proportion
LOWPASS     = 0.8
OU_TH, OU_SG = 0.2, 0.8
RADIUS      = 250
FEED_SEC    = 0.7
SHAPE_OFFSET_X = 150              # horizontal offset for central starting positions

# ───────────────────────────────────────────────────────
# 2  Donor-ID dialog
# ───────────────────────────────────────────────────────
expInfo = {"donor": ""}
gui.DlgFromDict(expInfo, title="Motion-trace donor ID")

# ───────────────────────────────────────────────────────
# 3  Window & stimuli
# ───────────────────────────────────────────────────────
win = visual.Window(fullscr=True, units="pix", color=[0.5]*3, allowGUI=False)
half_h = win.size[1] / 2
UI_X   = -(RADIUS + 40)
Y_TIMER = half_h - 40
Y_SCORE = half_h - 80

square = visual.Rect(win, 40, 40, fillColor="black", lineColor="black")
dot    = visual.Circle(win, 20, fillColor="black", lineColor="black")

timerTxt = visual.TextStim(win, pos=(UI_X, Y_TIMER),
                           color="white", height=24, alignText='left')
scoreTxt = visual.TextStim(win, pos=(UI_X, Y_SCORE),
                           color="white", height=24, alignText='left')
timerTxt.setAutoDraw(False)
scoreTxt.setAutoDraw(False)

# ───────────────────────────────────────────────────────
# 4  Helpers
# ───────────────────────────────────────────────────────
confine = lambda p, l=RADIUS: p if (r:=math.hypot(*p))<=l else (p[0]*l/r, p[1]*l/r)
def OU(n, th=OU_TH, sg=OU_SG, dt=1.0):
    v = np.zeros((n,2), np.float32)
    for t in range(1,n):
        v[t] = v[t-1] - th*v[t-1]*dt + sg*np.random.randn(2)*math.sqrt(dt)
    return v
fmt = lambda secs: f"{secs//60:02d}:{secs%60:02d}"

# ───────────────────────────────────────────────────────
# 5  Instructions
# ───────────────────────────────────────────────────────
visual.TextStim(
    win, color="white", height=32, wrapWidth=1600, alignText='center',
    text=(
        "Thank you for participating in our experiment.\n"
        "In the following game you will see two shapes whose movement will be .\n"
        "Move continuously for 3 s; then they freeze.\n"
        "Click the one you think you controlled!\n"
        "Press S to start."
    )
).draw()
win.flip()
if "escape" in event.waitKeys(keyList=["s","escape"]):
    win.close(); core.quit()

timerTxt.text = ""    # clear placeholders
scoreTxt.text = ""
timerTxt.setAutoDraw(True)
scoreTxt.setAutoDraw(True)

# ───────────────────────────────────────────────────────
# 6  Prepare central positions & counterbalance
# ───────────────────────────────────────────────────────
n_snips = DUR_FRAMES // SNIP_LEN           # should be 100 snippets
# half with square on left, half with square on right
positions = [True]*(n_snips//2) + [False]*(n_snips//2)
np.random.shuffle(positions)
snippet_count = 0

# ───────────────────────────────────────────────────────
# 7  Initial state
# ───────────────────────────────────────────────────────
mouse = event.Mouse(win=win, visible=True)
mouse.setPos((0,0))
control_shape = "square"
ou_vel = OU(int(DUR_FRAMES * 1.2))        # OU stream with headroom

vt = vd = np.zeros(2, np.float32)
score = 0
motion_frames = 0
trace = []                                # record raw positions

core.rush(True)
clk = core.Clock()

# ───────────────────────────────────────────────────────
# 8  Main loop
# ───────────────────────────────────────────────────────
while motion_frames < DUR_FRAMES:

    # ── wait for movement to initiate each snippet
    moved = False
    start_pos = mouse.getPos()
    while not moved:
        # draw in central starting pos for this snippet
        flag = positions[snippet_count]
        if flag:
            square.pos = (-SHAPE_OFFSET_X, 0)
            dot.pos    = ( SHAPE_OFFSET_X, 0)
        else:
            dot.pos    = (-SHAPE_OFFSET_X, 0)
            square.pos = ( SHAPE_OFFSET_X, 0)
        square.draw(); dot.draw(); win.flip()
        if event.getKeys(["escape"]):
            win.close(); core.quit()
        x,y = mouse.getPos()
        moved = math.hypot(x-start_pos[0], y-start_pos[1]) > 0.0

    seg_frames = 0

    # ── 3-second motion block ────────────────────────────
    while seg_frames < SNIP_LEN and motion_frames < DUR_FRAMES:
        t0 = clk.getTime()

        # update UI (timer based on moving frames)
        secs_left = (DUR_FRAMES - motion_frames) // FPS
        timerTxt.text = f"Time  {fmt(int(secs_left))}"
        scoreTxt.text = f"Score {score:03d}"

        # record & compute motion
        mx,my = mouse.getPos()
        trace.append((mx,my))
        if motion_frames == 0:
            dmx = dmy = 0
        else:
            dmx, dmy = np.array(trace[-1]) - np.array(trace[-2])

        odx, ody = ou_vel[motion_frames]
        mag_m, mag_o = math.hypot(dmx,dmy), math.hypot(odx,ody)
        if mag_o>0:
            odx, ody = odx*mag_m/mag_o, ody*mag_m/mag_o
        tdx = PROP*dmx + (1-PROP)*odx
        tdy = PROP*dmy + (1-PROP)*ody

        vt = LOWPASS*vt + (1-LOWPASS)*np.array([tdx,tdy])
        vd = LOWPASS*vd + (1-LOWPASS)*np.array([odx,ody])

        if control_shape=="square":
            square.pos = confine(square.pos + vt)
            dot.pos    = confine(dot.pos    + vd)
        else:
            dot.pos    = confine(dot.pos    + vt)
            square.pos = confine(square.pos + vd)

        square.draw(); dot.draw(); win.flip()

        seg_frames    += 1
        motion_frames += 1

        # maintain 60 Hz
        core.wait(max(0, DT - (clk.getTime() - t0)))

    # ── freeze & response ───────────────────────────────
    clicked = False
    while not clicked:
        square.draw(); dot.draw(); win.flip()
        if event.getKeys(["escape"]):
            win.close(); core.quit()
        if mouse.getPressed()[0]:
            pos = mouse.getPos()
            if square.contains(pos):
                clicked=True; choice="square"
            elif dot.contains(pos):
                clicked=True; choice="dot"
            core.wait(0.05)

    # feedback
    if choice == control_shape:
        score += 1
    good = square if control_shape=="square" else dot
    bad  = dot    if control_shape=="square" else square
    good.fillColor = "green"
    bad .fillColor = "red"
    square.draw(); dot.draw(); win.flip()
    core.wait(FEED_SEC)
    square.fillColor = dot.fillColor = "black"

    # swap control and advance snippet counter
    control_shape = "dot" if control_shape=="square" else "square"
    snippet_count += 1

core.rush(False)
win.close()

# ───────────────────────────────────────────────────────
# 9  Slice into snippets & compute features
# ───────────────────────────────────────────────────────
trace = np.array(trace, dtype=np.float32)
usable = (trace.shape[0] // SNIP_LEN) * SNIP_LEN
snips  = trace[:usable].reshape(-1, SNIP_LEN, 2)

Z = (snips - snips.mean(1,keepdims=True)) / (snips.std(1,keepdims=True)+1e-9)
diff   = np.diff(snips, axis=1)
dx, dy = diff[:,:,0], diff[:,:,1]
speed  = np.sqrt(dx**2 + dy**2)
theta  = np.arctan2(dy, dx)

features = np.column_stack([
    speed.mean(1),
    speed.std(1),
    np.unwrap(theta,axis=1).mean(1),
    np.unwrap(theta,axis=1).std(1),
    (speed.sum(1)/(np.linalg.norm(snips[:,-1]-snips[:,0],axis=1)+1e-9))[:,None],
    (speed<0.1).sum(1)/speed.shape[1]
]).astype(np.float32)

# ───────────────────────────────────────────────────────
# 10  Save to disk
# ───────────────────────────────────────────────────────
outdir = pathlib.Path(__file__).parent / "motion_donors"
outdir.mkdir(exist_ok=True)
ts   = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
base = f"donor_{expInfo['donor']}_{ts}"

np.save(outdir / f"{base}_snips.npy", Z.astype(np.float16))
np.save(outdir / f"{base}_feat.npy" , features)
print(f"Saved {Z.shape[0]} snippets for donor {expInfo['donor']} → {outdir}")

# ───────────────────────────────────────────────────────
# 11  Goodbye screen (rudimentary)
# ───────────────────────────────────────────────────────
win = visual.Window(fullscr=True, units="pix", color=[0.5]*3)
visual.TextStim(
    win,
    "Thank you for donating your motion traces!\n\n"
    "(Replace this text with your own goodbye screen.)",
    color="white", height=32, wrapWidth=1000, alignText='center'
).draw()
win.flip(); event.waitKeys()
win.close(); core.quit()