#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_motion_library_fullscreen.py   – v11
• Full-screen, black stimuli on grey.
• 50 % self-motion / 50 % OU noise.
• Controlled shape flips every 10 s; click response + colour feedback.
• Timer & score appear only after the participant presses **S** to start.
"""

# ────────────────────────────────────────────────────────
# 1 Imports
# ────────────────────────────────────────────────────────
import pathlib, datetime, math, numpy as np
from psychopy import visual, event, core, gui

# ────────────────────────────────────────────────────────
# 2 Parameters
# ────────────────────────────────────────────────────────
DUR_SEC, FPS    = 300, 60
FRAME_TOTAL     = DUR_SEC * FPS
SNIP_LEN        = 300
PROP            = 0.50
SWITCH_FRAMES   = 600
LOWPASS         = 0.8
OU_THETA, OU_SIGMA = 0.2, 0.8
CONF_RADIUS     = 250
FEED_SEC        = 0.7

# ────────────────────────────────────────────────────────
# 3 Donor ID dialog
# ────────────────────────────────────────────────────────
expInfo = {"donor": ""}
gui.DlgFromDict(expInfo, title="Motion-trace donor ID")

# ────────────────────────────────────────────────────────
# 4 Window & stimuli
# ────────────────────────────────────────────────────────
win = visual.Window(fullscr=True, units="pix", color=[0.5]*3, allowGUI=False)
half_w, half_h = win.size[0]/2, win.size[1]/2
UI_X       = -(CONF_RADIUS + 30)
UI_Y_TIMER =  half_h - 30
UI_Y_SCORE =  half_h - 70

square = visual.Rect(win, 40, 40, fillColor="black", lineColor="black")
dot    = visual.Circle(win, 20,  fillColor="black", lineColor="black")

timer_txt = visual.TextStim(win, pos=(UI_X, UI_Y_TIMER),
                            color="white", height=24, alignText='left')
score_txt = visual.TextStim(win, pos=(UI_X, UI_Y_SCORE),
                            color="white", height=24, alignText='left')
# keep them hidden until the task starts
timer_txt.setAutoDraw(False)
score_txt.setAutoDraw(False)

# ────────────────────────────────────────────────────────
# 5 Helpers
# ────────────────────────────────────────────────────────
confine = lambda p,l=CONF_RADIUS: p if (r:=math.hypot(*p))<=l else (p[0]*l/r,p[1]*l/r)
def OU(n,th=OU_THETA,sg=OU_SIGMA,dt=1):
    v=np.zeros((n,2),np.float32)
    for t in range(1,n): v[t]=v[t-1]-th*v[t-1]*dt+sg*np.random.randn(2)*math.sqrt(dt)
    return v
fmt = lambda s:f"{s//60:02d}:{s%60:02d}"

# ────────────────────────────────────────────────────────
# 6 Instruction screen
# ────────────────────────────────────────────────────────
visual.TextStim(
    win, color="white", height=32, wrapWidth=1600, alignText='center',
    text=(
        "Move the mouse continuously for 5 minutes.\n"
        "Every 10 s the shapes will PAUSE.\n"
        "Click the shape you believe you controlled:\n"
        "   ✓ CORRECT → GREEN  ✗ OTHER → RED\n"
        "Each correct click = +1 point.\n\n"
        "A countdown timer & score will appear\n"
        "in the upper-left corner once you start.\n\n"
        "Press **S** to START  ESC to quit."
    )
).draw()
win.flip()
if "escape" in event.waitKeys(keyList=["s","escape"]):
    win.close(); core.quit()

# ── enable overlays only AFTER start key ──────────────────
timer_txt.setAutoDraw(True)
score_txt.setAutoDraw(True)

# ────────────────────────────────────────────────────────
# 7 Initial placement & control assignment
# ────────────────────────────────────────────────────────
sqX,sqY=np.random.uniform(-200,200,2)
square.pos=(sqX,sqY); dot.pos=(-sqX,-sqY)
control_shape = np.random.choice(["square","dot"])

# ────────────────────────────────────────────────────────
# 8 Generate OU velocity stream
# ────────────────────────────────────────────────────────
ou_vel = OU(FRAME_TOTAL+1)

# ────────────────────────────────────────────────────────
# 9 Buffers & state
# ────────────────────────────────────────────────────────
trace=np.empty((FRAME_TOTAL,2),np.float32)
mouse=event.Mouse(win=win,visible=False); mouse.setPos((0,0))
vt=vd=np.zeros(2,np.float32); score=0; next_switch=SWITCH_FRAMES
clk=core.Clock(); core.rush(True)

# ────────────────────────────────────────────────────────
# 10 Main loop
# ────────────────────────────────────────────────────────
for f in range(FRAME_TOTAL):

    # update overlays
    secs_left=DUR_SEC-int(clk.getTime())
    timer_txt.text=f"Time  {fmt(secs_left)}"
    score_txt.text=f"Score {score:02d}"

    # mouse & OU
    mx,my=mouse.getPos(); trace[f]=(mx,my)
    dmx,dmy=(0,0) if f==0 else trace[f]-trace[f-1]
    odx,ody=ou_vel[f]; mag_m,mag_o=math.hypot(dmx,dmy),math.hypot(odx,ody)
    if mag_o>0: odx,ody=odx*mag_m/mag_o, ody*mag_m/mag_o
    tdx,tdy=PROP*dmx+(1-PROP)*odx, PROP*dmy+(1-PROP)*ody
    vt=LOWPASS*vt+(1-LOWPASS)*np.array([tdx,tdy])
    vd=LOWPASS*vd+(1-LOWPASS)*np.array([odx,ody])

    # move shapes
    if control_shape=="square":
        square.pos=confine(square.pos+vt); dot.pos=confine(dot.pos+vd)
    else:
        dot.pos=confine(dot.pos+vt); square.pos=confine(square.pos+vd)

    square.draw(); dot.draw(); win.flip()

    # pause & click
    if f+1==next_switch:
        mouse.setVisible(True); clicked=False
        while not clicked:
            secs_left=DUR_SEC-int(clk.getTime())
            timer_txt.text=f"Time  {fmt(secs_left)}"
            score_txt.text=f"Score {score:02d}"
            square.draw(); dot.draw(); win.flip()
            if "escape" in event.getKeys(): win.close(); core.quit()
            if mouse.getPressed()[0]:
                pos=mouse.getPos()
                if square.contains(pos): clicked=True; click="square"
                elif dot.contains(pos):  clicked=True; click="dot"
                core.wait(0.05)
        if click==control_shape: score+=1
        right=square if control_shape=="square" else dot
        wrong=dot    if control_shape=="square" else square
        right.fillColor="green"; wrong.fillColor="red"
        square.draw(); dot.draw(); win.flip(); core.wait(FEED_SEC)
        square.fillColor=dot.fillColor="black"; mouse.setVisible(False)
        control_shape="square" if control_shape=="dot" else "dot"
        next_switch+=SWITCH_FRAMES

    if "escape" in event.getKeys(): break

core.rush(False); win.close()

# ---------------- save snippets -------------------------
usable=(f//SNIP_LEN)*SNIP_LEN
snips=trace[:usable].reshape(-1,SNIP_LEN,2)
snips_z=(snips-snips.mean(1,keepdims=True))/(snips.std(1,keepdims=True)+1e-9)
outdir=pathlib.Path("motion_donors"); outdir.mkdir(exist_ok=True)
ts=datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
np.save(outdir/f"donor_{expInfo['donor']}_{ts}.npy",
        snips_z.astype(np.float16))
print("Saved", snips_z.shape[0], "snippets →", outdir)