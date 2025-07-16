import { core, util, visual } from './lib/psychojs-2024.2.4.js';
const { PsychoJS, Mouse } = core;
const { Scheduler } = util;

// --- Experiment Constants ---
const MAIN_TRIALS = 10;
const COMPLETE_URL = 'https://app.prolific.com/submissions/complete?cc=CHNYFRKK';

// --- Motion Dynamics Constants ---
const DURATION_SEC = 5;
const LOWPASS = 0.8;
const OU_TH = 0.2;
const OU_SG = 0.4;
const RADIUS = 250;
const OFFSET_X = 150;
const DRIFT_FREEZE_SEC = 0.0;
const FEED_SEC = 0.7;

const MOVEMENT_SCALING_FACTOR = 0.7;
const SELF_PROP = 0.60; // Fixed proportion for self-control

// --- Sampling Constants ---
const FPS = 60;
const SAMPLES_PER_SNIPPET = FPS * DURATION_SEC;
const SAMPLE_INTERVAL_MS = (DURATION_SEC * 1000) / SAMPLES_PER_SNIPPET;

// --- Noise Generation Constants ---
const NOISE_RATE_HZ = 20;
const NOISE_INTERVAL = Math.round(FPS / NOISE_RATE_HZ);

// --- Global State Variables ---
let selectedShape = 'square';
let ouVelFull;
let ouVelCoarse;
let previousTrialMouseTrajectory = null; // To store mouse data from the previous trial

let targetOuIndex = 0;
let distractorCoarseBase = 0;

let win, mouse, square, dot, choiceTxt;
let phase, snippet, control, fbUntil;
let vt = [0, 0], vd = [0, 0], positions, trace = [];
let newTrial = false, waitStart = [0, 0], moved = false;
let timeSampler = null;

// --- Utility Functions ---
const randn = () => Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
const OU = n => {
    const v = Array.from({ length: n }, () => [0, 0]);
    for (let t = 1; t < n; t++) {
        const [px, py] = v[t - 1];
        v[t] = [
            px - OU_TH * px + OU_SG * randn(),
            py - OU_TH * py + OU_SG * randn()
        ];
    }
    return v;
};
const conf = p => {
    const r = Math.hypot(p[0], p[1]);
    return r <= RADIUS ? p : [p[0] * RADIUS / r, p[1] * RADIUS / r];
};

// --- Core Classes ---

class TimeSampler {
    constructor(durationMs, samplesTarget) {
        this.durationMs = durationMs;
        this.samplesTarget = samplesTarget;
        this.sampleInterval = durationMs / samplesTarget;
        this.samples = [];
        this.startTime = null;
        this.interpolationBuffer = [];
    }
    start() {
        this.samples = [];
        this.startTime = performance.now();
        this.interpolationBuffer = [];
    }
    update(currentPos) {
        if (!this.startTime) return false;
        const now = performance.now();
        const elapsed = now - this.startTime;
        this.interpolationBuffer.push({ time: now, pos: [...currentPos] });
        this.interpolationBuffer = this.interpolationBuffer.filter(e => now - e.time < 100);
        while (this.samples.length < this.samplesTarget &&
               (this.samples.length * this.sampleInterval) <= elapsed) {
            const targetTime = this.startTime + (this.samples.length * this.sampleInterval);
            const interpolatedPos = this.interpolatePosition(targetTime);
            this.samples.push({
                time: targetTime - this.startTime,
                x: interpolatedPos[0],
                y: interpolatedPos[1],
                sampleIndex: this.samples.length
            });
        }
        return elapsed >= this.durationMs || this.samples.length >= this.samplesTarget;
    }
    interpolatePosition(targetTime) {
        if (this.interpolationBuffer.length === 0) return [0, 0];
        if (this.interpolationBuffer.length === 1) return [...this.interpolationBuffer[0].pos];
        let before = this.interpolationBuffer[0];
        let after = this.interpolationBuffer[this.interpolationBuffer.length - 1];
        for (let i = 0; i < this.interpolationBuffer.length - 1; i++) {
            if (this.interpolationBuffer[i].time <= targetTime &&
                this.interpolationBuffer[i + 1].time >= targetTime) {
                before = this.interpolationBuffer[i];
                after = this.interpolationBuffer[i + 1];
                break;
            }
        }
        if (after.time === before.time) return before.pos;
        const t = (targetTime - before.time) / (after.time - before.time);
        return [
            before.pos[0] + t * (after.pos[0] - before.pos[0]),
            before.pos[1] + t * (after.pos[1] - before.pos[1])
        ];
    }
    getSamples() {
        while (this.samples.length < this.samplesTarget) {
            const lastSample = this.samples[this.samples.length - 1] || { x: 0, y: 0 };
            this.samples.push({
                time: this.samples.length * this.sampleInterval,
                x: lastSample.x,
                y: lastSample.y,
                sampleIndex: this.samples.length
            });
        }
        return this.samples.slice(0, this.samplesTarget);
    }
}

class BulletproofCursorHider {
    constructor() {
        this.virtualMouseX = 0;
        this.virtualMouseY = 0;
        this.pointerLocked = false;
        this.isSupported = 'pointerLockElement' in document;
        this.movementThreshold = 300;
        this.retryAttempts = 3;
        this.retryCount = 0;
        this.callbacks = { onMovement: [] };
        this.init();
    }
    init() {
        this.addGlobalCSS();
        if (this.isSupported) this.setupPointerLock();
        this.applyBrowserFixes();
        this.setupFallbackHiding();
    }
    addGlobalCSS() {
        const style = document.createElement('style');
        style.textContent = `
            .cursor-hidden, .cursor-hidden * { cursor: none !important; }
            html, body, canvas, div, span { cursor: none !important; }
            .psychojs-window, .psychojs-window * { cursor: none !important; }
            body { overflow: hidden !important; }
            .chrome-fix { cursor: url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAFfETKV3QAAAABJRU5ErkJggg=='), none !important; }
        `;
        document.head.appendChild(style);
    }
    setupPointerLock() {
        document.addEventListener('pointerlockchange', () => this.handleLockChange());
        document.addEventListener('pointerlockerror', e => this.handleLockError(e));
        this.boundMouseMove = e => this.handlePointerLockMovement(e);
    }
    async requestPointerLock() {
        if (this.pointerLocked) return;
        const element = document.documentElement;
        try {
            await element.requestPointerLock({ unadjustedMovement: true });
            this.retryCount = 0;
        } catch {
            try {
                await element.requestPointerLock();
                this.retryCount = 0;
            } catch {
                if (this.retryCount < this.retryAttempts) {
                    this.retryCount++;
                    setTimeout(() => this.requestPointerLock(), 1000);
                } else {
                    this.setupFallbackHiding();
                }
            }
        }
    }
    handleLockChange() {
        const wasLocked = this.pointerLocked;
        this.pointerLocked = document.pointerLockElement === document.documentElement;
        if (this.pointerLocked && !wasLocked) {
            document.addEventListener('mousemove', this.boundMouseMove);
        } else if (!this.pointerLocked && wasLocked) {
            document.removeEventListener('mousemove', this.boundMouseMove);
            this.applyCSSHiding();
        }
    }
    handleLockError() {
        this.setupFallbackHiding();
    }
    handlePointerLockMovement(event) {
        if (Math.abs(event.movementX) > this.movementThreshold ||
            Math.abs(event.movementY) > this.movementThreshold) return;
        this.virtualMouseX += event.movementX;
        this.virtualMouseY += event.movementY;
        const maxX = window.innerWidth / 2;
        const maxY = window.innerHeight / 2;
        this.virtualMouseX = Math.max(-maxX, Math.min(maxX, this.virtualMouseX));
        this.virtualMouseY = Math.max(-maxY, Math.min(maxY, this.virtualMouseY));
        this.emit('movement', { x: this.virtualMouseX, y: this.virtualMouseY });
    }
    setupFallbackHiding() {
        this.applyCSSHiding();
        ['mouseenter', 'mousemove', 'focus', 'click'].forEach(event => {
            document.addEventListener(event, () => this.applyCSSHiding(), { passive: true });
        });
        document.addEventListener('keydown', () => {
            setTimeout(() => this.applyCSSHiding(), 0);
        });
    }
    applyCSSHiding() {
        document.documentElement.classList.add('cursor-hidden');
        document.body.classList.add('cursor-hidden');
    }
    applyBrowserFixes() {
        const isChrome = /Chrome/.test(navigator.userAgent);
        const isWindows = /Windows/.test(navigator.userAgent);
        if (isChrome && isWindows) document.body.classList.add('chrome-fix');
    }
    on(event, callback) {
        if (this.callbacks[event]) this.callbacks[event].push(callback);
    }
    emit(event, data) {
        if (this.callbacks[event]) this.callbacks[event].forEach(cb => cb(data));
    }
    getMousePosition() {
        if (this.pointerLocked) return [this.virtualMouseX, this.virtualMouseY];
        return mouse ? mouse.getPos() : [0, 0];
    }
    setInitialPosition(x, y) {
        this.virtualMouseX = x;
        this.virtualMouseY = y;
    }
}


// --- PsychoJS Setup ---
const psychoJS = new PsychoJS({ debug: false });
psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.ERROR);

const params = new URLSearchParams(window.location.search);
const prolificID = params.get('PROLIFIC_PID') || '';
const expInfo = { Participant: prolificID };

psychoJS.openWindow({ fullscr: true, color: new util.Color([0.5, 0.5, 0.5]), units: 'pix' });

const cursorHider = new BulletproofCursorHider();

const flow = new Scheduler(psychoJS);
const cancel = new Scheduler(psychoJS);

if (prolificID) {
  psychoJS.schedule(flow);
} else {
    psychoJS.schedule(psychoJS.gui.DlgFromDict({ dictionary: expInfo, title: 'motionlibrary' }));
    psychoJS.scheduleCondition(() => psychoJS.gui.dialogComponent.button === 'OK', flow, cancel);
}

// --- Experiment Flow ---
flow.add(expInit);
flow.add(instrMainBegin); flow.add(instrMainLoop);
flow.add(recBegin); flow.add(recLoop); flow.add(recEnd);
flow.add(byeBegin); flow.add(byeLoop); flow.add(byeEnd);
flow.add(quit, '', true); cancel.add(quit, '', false);

psychoJS.start({ expName: 'motionlibrary', expInfo });

// --- Routine Implementations ---

function expInit() {
    win = psychoJS.window;
    mouse = new Mouse({ win, visible: false });
    cursorHider.setInitialPosition(0, 0);
    setTimeout(() => cursorHider.requestPointerLock(), 500);

    square = new visual.Rect({ win, width: 40, height: 40, fillColor: 'black', lineColor: 'black' });
    const verts = Array.from({ length: 60 }, (_, i) => [Math.cos(2 * Math.PI * i / 60) * 20, Math.sin(2 * Math.PI * i / 60) * 20]);
    dot = new visual.ShapeStim({ win, vertices: verts, fillColor: 'black', lineColor: 'black' });

    const half = win.size[1] / 2;
    choiceTxt = new visual.TextStim({ win, pos: [0, half - 150], height: 32, color: 'yellow', alignHoriz: 'center', text: 'Which one did you control more?\n(← or →)' });

    const totalFrames = MAIN_TRIALS * SAMPLES_PER_SNIPPET * 2;
    ouVelFull = OU(totalFrames);
    const coarseLength = Math.ceil(totalFrames / NOISE_INTERVAL) + 1;
    ouVelCoarse = OU(coarseLength);

  return Scheduler.Event.NEXT;
}

let instr;
function instrMainBegin() {
    cursorHider.requestPointerLock();
    instr = new visual.TextStim({ win, height: 32, color: 'white', wrapWidth: 1600, alignHoriz: 'center',
        text: `Welcome to the experiment!\n\nIn this experiment, you will see a SQUARE and a CIRCLE.\n\nYour task is to move your mouse and determine which shape\nyou have more control over.\n\nOne shape will follow your mouse movements more closely,\nwhile the other will move more independently.\n\nAfter each trial, use the ARROW KEYS (← →) to choose\nthe shape you felt you controlled more.\n\nThe cursor will be hidden throughout the experiment.\nThere will be ${MAIN_TRIALS} trials in total.\n\nPress S to begin the experiment.` });
    instr.setAutoDraw(true);
  return Scheduler.Event.NEXT;
}

function instrMainLoop() {
    if (psychoJS.eventManager.getKeys({ keyList: ['s'] }).length) {
        instr.setAutoDraw(false);
    return Scheduler.Event.NEXT;
  }
  return Scheduler.Event.FLIP_REPEAT;
}

function recBegin() {
    phase = 'WAIT_MOVE';
    snippet = 0;
    control = 'square';
    vt = [0, 0];
    vd = [0, 0];
    trace = [];
    newTrial = true;
    moved = false;
    previousTrialMouseTrajectory = null;
    selectedShape = 'square';

    positions = [...Array(Math.ceil(MAIN_TRIALS / 2)).fill(true), ...Array(Math.floor(MAIN_TRIALS / 2)).fill(false)];
    util.shuffle(positions);

    square.setAutoDraw(true);
    dot.setAutoDraw(true);

    if (!cursorHider.pointerLocked) cursorHider.requestPointerLock();

    return Scheduler.Event.NEXT;
}

function recLoop() {
    const [mx, my] = cursorHider.getMousePosition();

    switch (phase) {
        case 'WAIT_MOVE': {
            if (newTrial) {
                waitStart = [mx, my];
                moved = false;
                newTrial = false;
                timeSampler = new TimeSampler(DURATION_SEC * 1000, SAMPLES_PER_SNIPPET);
                targetOuIndex = Math.floor(Math.random() * (ouVelFull.length - SAMPLES_PER_SNIPPET));
                distractorCoarseBase = Math.floor(Math.random() * (ouVelCoarse.length -
                    Math.ceil(SAMPLES_PER_SNIPPET / NOISE_INTERVAL) - 1));
            }
            const left = positions[snippet];
            square.setPos(left ? [-OFFSET_X, 0] : [OFFSET_X, 0]);
            dot.setPos(left ? [OFFSET_X, 0] : [-OFFSET_X, 0]);

            if (!moved && Math.hypot(mx - waitStart[0], my - waitStart[1]) > 0) {
                moved = true;
                vt = [0, 0];
                vd = [0, 0];
                trace = [];
                timeSampler.start();
                phase = 'MOVE';
      }
            break;
        }
        case 'MOVE': {
            const samplingComplete = timeSampler.update([mx, my]);
            const now = performance.now();
            const elapsed = Math.max(0, now - timeSampler.startTime);
            const currentSampleIndex = Math.min(Math.floor(elapsed / SAMPLE_INTERVAL_MS), SAMPLES_PER_SNIPPET - 1);

            let dmx = trace.length > 0 ? mx - trace[trace.length - 1][0] : 0;
            let dmy = trace.length > 0 ? my - trace[trace.length - 1][1] : 0;
            dmx *= MOVEMENT_SCALING_FACTOR;
            dmy *= MOVEMENT_SCALING_FACTOR;

            let [odx_target, ody_target] = ouVelFull[targetOuIndex + currentSampleIndex];

            let odx_distractor = 0, ody_distractor = 0;

            if (snippet === 0 || !previousTrialMouseTrajectory) {
                // For the first trial, use OU noise for the distractor
                const coarseIdx = Math.floor(currentSampleIndex / NOISE_INTERVAL);
                const alpha = (currentSampleIndex % NOISE_INTERVAL) / NOISE_INTERVAL;
                let [nx0, ny0] = ouVelCoarse[distractorCoarseBase + coarseIdx];
                let [nx1, ny1] = ouVelCoarse[distractorCoarseBase + coarseIdx + 1];
                odx_distractor = nx0 + alpha * (nx1 - nx0);
                ody_distractor = ny0 + alpha * (ny1 - ny0);
            } else {
                // For subsequent trials, use the previous trial's recorded mouse movement
                if (currentSampleIndex > 0) {
                    const prevSample = previousTrialMouseTrajectory[currentSampleIndex - 1];
                    const currentSample = previousTrialMouseTrajectory[currentSampleIndex];
                    odx_distractor = currentSample.x - prevSample.x;
                    ody_distractor = currentSample.y - prevSample.y;
                }
            }

            // --- Velocity Scaling ---
            const mM = Math.hypot(dmx, dmy);
            const oM_target = Math.hypot(odx_target, ody_target);
            if (oM_target > 0) {
                const r = mM / oM_target;
                odx_target *= r;
                ody_target *= r;
            }
            const oM_dist = Math.hypot(odx_distractor, ody_distractor);
            if (oM_dist > 0) {
                const r = mM / oM_dist;
                odx_distractor *= r;
                ody_distractor *= r;
            }

            // --- Low-pass Filtering ---
            vt[0] = LOWPASS * vt[0] + (1 - LOWPASS) * (SELF_PROP * dmx + (1 - SELF_PROP) * odx_target);
            vt[1] = LOWPASS * vt[1] + (1 - LOWPASS) * (SELF_PROP * dmy + (1 - SELF_PROP) * ody_target);
            vd[0] = LOWPASS * vd[0] + (1 - LOWPASS) * odx_distractor;
            vd[1] = LOWPASS * vd[1] + (1 - LOWPASS) * ody_distractor;

            trace.push([mx, my]);

            if (control === 'square') {
                square.setPos(conf([square.pos[0] + vt[0], square.pos[1] + vt[1]]));
                dot.setPos(conf([dot.pos[0] + vd[0], dot.pos[1] + vd[1]]));
      } else {
                dot.setPos(conf([dot.pos[0] + vt[0], dot.pos[1] + vt[1]]));
                square.setPos(conf([square.pos[0] + vd[0], square.pos[1] + vd[1]]));
      }

            if (samplingComplete) {
                previousTrialMouseTrajectory = timeSampler.getSamples(); // Save trajectory for the next trial

                previousTrialMouseTrajectory.forEach((sample, idx) => {
                    psychoJS.experiment.addData('trial', snippet);
                    psychoJS.experiment.addData('sampleIndex', idx);
                    psychoJS.experiment.addData('sampleTime', sample.time);
                    psychoJS.experiment.addData('x', sample.x);
                    psychoJS.experiment.addData('y', sample.y);
                    psychoJS.experiment.addData('selfProp', SELF_PROP);
          psychoJS.experiment.nextEntry();
                });

                fbUntil = performance.now() + DRIFT_FREEZE_SEC * 1000;
                phase = 'FREEZE_DRIFT';
      }
            break;
        }
        case 'FREEZE_DRIFT': {
            if (performance.now() >= fbUntil) {
                const left = positions[snippet];
                square.setPos(left ? [-OFFSET_X, 0] : [OFFSET_X, 0]);
                dot.setPos(left ? [OFFSET_X, 0] : [-OFFSET_X, 0]);
                choiceTxt.setAutoDraw(true);
                phase = 'CHOICE';
      }
            break;
        }
        case 'CHOICE': {
            const keys = psychoJS.eventManager.getKeys({ keyList: ['left', 'right'] });
            let choiceMade = null;
            if (keys.length > 0) {
                const isSquareOnLeft = positions[snippet];
                const leftShapeName = isSquareOnLeft ? 'square' : 'dot';
                const rightShapeName = isSquareOnLeft ? 'dot' : 'square';
                if (keys.includes('left')) choiceMade = leftShapeName;
                else if (keys.includes('right')) choiceMade = rightShapeName;
            }

            if (choiceMade) {
                selectedShape = choiceMade;
                const correct = selectedShape === control;

                (control === 'square' ? square : dot).fillColor = 'green';
                (control === 'square' ? dot : square).fillColor = 'red';

          choiceTxt.setAutoDraw(false);
                fbUntil = performance.now() + FEED_SEC * 1000;
                phase = 'FEEDBACK';
        }
            break;
        }
        case 'FEEDBACK': {
            if (performance.now() >= fbUntil) {
                square.fillColor = 'black';
                dot.fillColor = 'black';
                control = control === 'square' ? 'dot' : 'square';
                snippet++;
                newTrial = true;
                phase = snippet >= MAIN_TRIALS ? 'DONE' : 'WAIT_MOVE';
      }
            break;
        }
        case 'DONE':
      return Scheduler.Event.NEXT;
  }

    if (psychoJS.eventManager.getKeys({ keyList: ['escape'] }).length) return quit();
  return Scheduler.Event.FLIP_REPEAT;
}

function recEnd() {
    square.setAutoDraw(false);
    dot.setAutoDraw(false);
    choiceTxt.setAutoDraw(false);

    psychoJS.experiment.addData('participant_id', prolificID);
    psychoJS.experiment.addData('cursorHidingMethod', cursorHider.pointerLocked ? 'PointerLock' : 'CSS');
    psychoJS.experiment.addData('samplesPerSnippet', SAMPLES_PER_SNIPPET);
    psychoJS.experiment.nextEntry();

    return Scheduler.Event.NEXT;
  }

let bye, byeTimer;
function byeBegin() {
    bye = new visual.TextStim({ win, height: 32, color: 'white',
        text: `Thank you for participating!\n\nYour responses have been saved.\nYou will be redirected to Prolific in a moment...` });
  bye.setAutoDraw(true);
    byeTimer = new util.CountdownTimer(5);
  return Scheduler.Event.NEXT;
}

function byeLoop() {
    if (byeTimer.getTime() <= 0) return Scheduler.Event.NEXT;
    return Scheduler.Event.FLIP_REPEAT;
}

function byeEnd() {
    bye.setAutoDraw(false);
    return Scheduler.Event.NEXT;
}

function quit(msg = '', done = false) {
    psychoJS.quit({ message: msg, isCompleted: done }).then(() => {
        if (done && prolificID) window.location.href = COMPLETE_URL;
  });
  return Scheduler.Event.QUIT;
}
