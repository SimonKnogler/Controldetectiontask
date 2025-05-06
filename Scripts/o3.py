#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Control‑Detection Task — cleaned‑up v2.2  (2025‑05‑05)
------------------------------------------------------
• Toggle `SIMULATE` at the top (True = autorun, False = real run).
• Uses *self‑motion proportion* (`prop`) so higher numbers = easier control.
• Staircases start at 0.50 (low cue) and 0.80 (high cue) self‑motion.
• “Medium” proportion computed separately per MODE as the post‑staircase mean.
• Trial length = 3 s. 500 ms blank before agency rating.
• Confidence legend matches proposal wording.
"""

# ====================================================
# 0  Imports & SIMULATE toggle  🔧
# ====================================================
import os, sys, math, random, pathlib, datetime
import numpy as np
from psychopy import visual, event, core, data, gui

# ── Toggle this line only ───────────────────────────
SIMULATE = False      # True = autorun, False = full experiment
# ───────────────────────────────────────────────────

if SIMULATE:
    print("⚙ Autorun mode ON – running without a real window.")
    class _DummyWin:
        size = (1920,1080)
        def __getattr__(self, name):
            def _(*a,**k): return None
            return _
    visual.Window = lambda *a,**k: _DummyWin()
    rng = np.random.default_rng(42)
    event.waitKeys = lambda keyList=None, **k: [keyList[int(rng.integers(len(keyList)))] ] if keyList else []
    class _FakeMouse:
        def __init__(self,*a,**k): self.pos=np.zeros(2)
        def setPos(self,p): self.pos[:]=0
        def getPos(self): self.pos+=rng.normal(scale=3,size=2); return tuple(self.pos)
    event.Mouse = _FakeMouse

# ====================================================
# 1  Experiment Info & counter‑balancing
# ====================================================
expName = "ControlDetection_WP1_v2"
expInfo = {"participant":"","session":"001"}
if not SIMULATE:
    if not gui.DlgFromDict(expInfo, expName).OK: core.quit()
else:
    expInfo["participant"]="SIM"

if random.random()<0.5:
    low_col, high_col = "blue","green"
else:
    low_col, high_col = "green","blue"
expInfo.update({"low_precision_colour":low_col,"high_precision_colour":high_col})

# ====================================================
# 2  Paths & ExperimentHandler
# ====================================================
root = pathlib.Path.cwd()/"data"; root.mkdir(exist_ok=True)
file_stem=f"{expName}_P{expInfo['participant']}_S{expInfo['session']}_{datetime.datetime.now():%Y%m%dT%H%M%S}"
csv_path  = (root / f"{file_stem}.csv") 
thisExp=data.ExperimentHandler(name=expName,extraInfo=expInfo,savePickle=False,saveWideText=False,dataFileName=root/file_stem)

# ====================================================
# 3  Window & global clock
# ====================================================
win=visual.Window(size=(1920,1080),fullscr=not SIMULATE,color=[0.5]*3,units="pix")
globalClock=core.Clock()

# ====================================================
# 4  Stimuli
# ====================================================
square=visual.Rect(win,width=40,height=40,fillColor="black",lineColor="black")
dot   =visual.Circle(win,radius=20,fillColor="black",lineColor="black")
fix   =visual.TextStim(win,text="+",color="white",height=30)
msg   =visual.TextStim(win,text="",color="white",height=26,wrapWidth=1000)

# ====================================================
# 5  Helpers
# ====================================================
confine=lambda p,l=250:(p if (r:=math.hypot(*p))<=l else (p[0]*l/r,p[1]*l/r))
rotate=lambda vx,vy,a:(vx*math.cos(math.radians(a))-vy*math.sin(math.radians(a)),
                      vx*math.sin(math.radians(a))+vy*math.cos(math.radians(a)))

def OU_snippet(n=300,theta=.2,sigma=.8,dt=1.,lin_bias=(0,0),ang_bias=0.):
    lin_bias=np.asarray(lin_bias,float)
    v=np.zeros((n,2))
    for t in range(1,n):
        v[t]=v[t-1]-theta*v[t-1]*dt+sigma*np.random.randn(2)*math.sqrt(dt)+lin_bias*dt
        spd=math.hypot(*v[t]); ang=math.atan2(v[t,1],v[t,0])+ang_bias*dt
        v[t]=spd*np.array([math.cos(ang),math.sin(ang)])
    return v

# ====================================================
# 6  Pre‑experiment demo (bias estimate)
# ====================================================
mouse=event.Mouse(win=win,visible=not SIMULATE); mouse.setPos((0,0))
msg.text="30‑s DEMO – move the mouse; one shape will mostly follow you.\nPress any key."; msg.draw(); win.flip(); event.waitKeys()
fix.draw(); win.flip(); core.wait(.5)

sqX,sqY=np.random.uniform(-200,200,2); dotX,dotY=-sqX,-sqY
square.pos,dot.pos=(sqX,sqY),(dotX,dotY); control_shape=random.choice(["square","dot"])

traj=[]; clk=core.Clock(); last=mouse.getPos(); frame=0; snip=OU_snippet(1800)
while clk.getTime()<30:
    x,y=mouse.getPos(); dx,dy=x-last[0],y-last[1]; last=(x,y)
    ou_dx,ou_dy=snip[frame%1800]; frame+=1
    traj.append([dx,dy])
    if control_shape=="square":
        square.pos=confine((square.pos[0]+dx,square.pos[1]+dy)); dot.pos=confine((dot.pos[0]+ou_dx,dot.pos[1]+ou_dy))
    else:
        dot.pos=confine((dot.pos[0]+dx,dot.pos[1]+dy)); square.pos=confine((square.pos[0]+ou_dx,square.pos[1]+ou_dy))
    square.draw(); dot.draw(); win.flip()
traj=np.array(traj); lin_bias=traj.mean(0); ang_bias=np.nanmean(np.diff(np.arctan2(traj[:,1],traj[:,0]))) if traj.size else 0.

# ====================================================
# 7  OU library
# ====================================================
LIB_N,LIB_LEN=1000,300; ou_lib=np.array([OU_snippet(LIB_LEN,lin_bias=lin_bias,ang_bias=ang_bias) for _ in range(LIB_N)])

# ====================================================
# 8  Staircases (mode × expectation)
# ====================================================
MODES=[0,90]; EXPECT=["low","high"]
PROP={m:{"low":0.50,"high":0.80} for m in MODES}; TARGET={"low":0.60,"high":0.90}; STEP=0.05
COUNTS={m:{e:[0,0] for e in EXPECT} for m in MODES}

# ====================================================
# 9  Trial helper
# ====================================================
BREAK_EVERY=3; LOWPASS=0.8
CONF_KEYS=["1","2","3","4"]
SHAPE_FROM_KEY={"1":"square","2":"square","3":"dot","4":"dot"}
CONF_FROM_KEY={k:i+1 for i,k in enumerate(CONF_KEYS)}

MEDIUM_PROP_MODE={m:(PROP[m]["low"]+PROP[m]["high"])/2 for m in MODES}  # placeholder, will be recomputed

def run_trial(phase,angle_bias,expect_level,true_or_med):
    prop=PROP[angle_bias][expect_level] if true_or_med=="true" else MEDIUM_PROP_MODE[angle_bias]
    cue_col=low_col if expect_level=="low" else high_col
    for stim in (fix,square,dot): stim.color=cue_col if hasattr(stim,'color') else None
    fix.draw(); win.flip(); core.wait(.5)
    sqX,sqY=np.random.uniform(-200,200,2); dotX,dotY=-sqX,-sqY; square.pos,dot.pos=(sqX,sqY),(dotX,dotY)
    target_shape=random.choice(["square","dot"]); snip=ou_lib[random.randrange(LIB_N)]
    mouse.setPos((0,0)); last=(0,0); vt=vd=np.zeros(2); clk=core.Clock(); frame=0
    while clk.getTime()<3.0:
        x,y=mouse.getPos(); dx,dy=x-last[0],y-last[1]; last=(x,y); dx,dy=rotate(dx,dy,angle_bias)
        ou_dx,ou_dy=snip[frame%LIB_LEN]; frame+=1
        mag_m,mag_o=math.hypot(dx,dy),math.hypot(ou_dx,ou_dy)
        if mag_o>0: ou_dx,ou_dy=ou_dx/mag_o*mag_m, ou_dy/mag_o*mag_m
        targ_dx,targ_dy=prop*dx+(1-prop)*ou_dx, prop*dy+(1-prop)*ou_dy
        dist_dx,dist_dy=ou_dx,ou_dy
        vt=LOWPASS*vt+(1-LOWPASS)*np.array([targ_dx,targ_dy]); vd=LOWPASS*vd+(1-LOWPASS)*np.array([dist_dx,dist_dy])
        if target_shape=="square":
            square.pos=confine(square.pos+vt); dot.pos=confine(dot.pos+vd)
        else:
            dot.pos=confine(dot.pos+vt); square.pos=confine(square.pos+vd)
        square.draw(); dot.draw(); win.flip()
    msg.text=("Which shape did you control & how confident?\n"
              "1 Square (guess)   2 Square (confident)\n"
              "3 Dot (confident)  4 Dot (guess)")
    msg.draw(); win.flip(); key=event.waitKeys(keyList=CONF_KEYS+["escape"])[0]
    if key=="escape": core.quit()
    resp_shape,conf_lvl=SHAPE_FROM_KEY[key],CONF_FROM_KEY[key]
    correct=int(resp_shape==target_shape)
    core.wait(0.5)  # blank before rating
    rating=random.randint(0,100) if SIMULATE else 0
    if not SIMULATE:
        msg.text="How much control did you feel? (0‑100, Enter)"; msg.draw(); win.flip(); txt=""; done=False
        while not done:
            for k in event.waitKeys():
                if k in ("return","enter"): done=True
                elif k=="backspace": txt=txt[:-1]
                elif k.isdigit() and len(txt)<3: txt+=k
                elif k=="escape": core.quit()
            msg.text=f"How much control? (0‑100)\nCurrent: {txt}"; msg.draw(); win.flip()
        rating=float(txt) if txt else 0
    if phase=="practice" or true_or_med=="true":
        cOK,cTOT=COUNTS[angle_bias][expect_level]; COUNTS[angle_bias][expect_level]=[cOK+correct,cTOT+1]
        acc=(cOK+correct)/(cTOT+1); tgt=TARGET[expect_level]
        if acc>tgt: PROP[angle_bias][expect_level]=max(0,PROP[angle_bias][expect_level]-STEP)
        else:       PROP[angle_bias][expect_level]=min(1,PROP[angle_bias][expect_level]+STEP)
    return {"phase":phase,"angle_bias":angle_bias,"expect_level":expect_level,"true_or_med":true_or_med,
            "true_shape":target_shape,"resp_shape":resp_shape,"conf_level":conf_lvl,"accuracy":correct,
            "control_rating":rating,"prop_used":prop}

# ====================================================
# 10  Practice block (60 trials)
# ====================================================
PRACTICE_PER_CELL=3
conds=[(m,e) for m in MODES for e in EXPECT]*PRACTICE_PER_CELL; random.shuffle(conds)
msg.text="PRACTICE – ≈5 min.\nPress any key."; msg.draw(); win.flip(); event.waitKeys()
for i,(m,e) in enumerate(conds,1):
    res=run_trial("practice",m,e,"true"); [thisExp.addData(k,v) for k,v in res.items()]; thisExp.nextEntry()
    if i%30==0 and not SIMULATE:
        msg.text="Short break – press a key."; msg.draw(); win.flip(); event.waitKeys()

# recompute medium proportions per mode
MEDIUM_PROP_MODE={m:(PROP[m]["low"]+PROP[m]["high"])/2 for m in MODES}


# ====================================================
# 11. Test block  (75 medium trials per cell  → 300)
# ====================================================
MEDIUM_PER_CELL = 2

msg.text = "MAIN BLOCK – about 25 min.\nPress any key to start."; msg.draw(); win.flip(); event.waitKeys()

test_conditions = [(m,e) for m in MODES for e in EXPECT]*MEDIUM_PER_CELL
random.shuffle(test_conditions)

trial_counter = 0
for m,e in test_conditions:
    trial_counter += 1
    res = run_trial("test", m, e, "medium")
    for k,v in res.items(): thisExp.addData(k,v)
    thisExp.nextEntry()
    if (trial_counter % BREAK_EVERY)==0 and not SIMULATE:
        msg.text = "Break – you may rest for 30 s.  Continue whenever ready."; msg.draw(); win.flip(); core.wait(30)

# ====================================================
# 12. Save & goodbye
# ====================================================
file_csv = root_path / f"{file_stem}.csv"
thisExp.saveAsWideText(file_csv)
print(f"Data saved to {file_csv}")

msg.text = "Thank you – task complete!  Press any key."; msg.draw(); win.flip(); event.waitKeys()
win.close(); core.quit()
