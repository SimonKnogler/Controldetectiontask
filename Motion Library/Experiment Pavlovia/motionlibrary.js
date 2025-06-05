/*******************************************************************************
 *  Motion-trace task – 2-down / 1-up staircase (≈60 % correct)               *
 *  • Practice: 10 trials; staircase adapts                                    *
 *  • Main:     10 trials at calibrated PROP                                   *
 *  • One CSV on Pavlovia: frame rows + staircase rows                         *
 *  • Auto-redirect to Prolific on completion                                  *
 ******************************************************************************/

import { core, util, visual } from './lib/psychojs-2024.2.4.js';
const { PsychoJS, Mouse } = core;
const { Scheduler }       = util;

/* ─────────────────── CONSTANTS ─────────────────── */
const FPS  = 60, SNIP_LEN = 180;
const PRACTICE_TRIALS = 2, MAIN_TRIALS = 2;
const PRACTICE_FRAMES = PRACTICE_TRIALS*SNIP_LEN;
const MAIN_FRAMES     = MAIN_TRIALS   *SNIP_LEN;
const LOWPASS=0.8, OU_TH=0.2, OU_SG=0.8;
const RADIUS=250, FEED_SEC=0.7, OFFSET_X=150, DRIFT_FREEZE_SEC=1.0;
const STEP=0.05, MIN_PROP=0.20, MAX_PROP=0.90;
const COMPLETE_URL = 'https://app.prolific.com/submissions/complete?cc=CHNYFRKK';

/* ─────────────────── GLOBAL VARS ──────────────── */
let selfProp=0.6, consecutiveCorrect=0, consecutiveErrors=0;
let pathLens=[], minMove=null;

/* ─────────────────── INIT PSYCHOJS ────────────── */
const psychoJS = new PsychoJS({ debug:false });
psychoJS.experimentLogger.setLevel(core.Logger.ServerLevel.ERROR);

/* participant ID from Prolific URL */
const params = new URLSearchParams(window.location.search);
const prolificID = params.get('PROLIFIC_PID') || '';

const expInfo = { Participant: prolificID };

/* open window */
psychoJS.openWindow({
  fullscr:true, color:new util.Color([0.5,0.5,0.5]), units:'pix'
});

/* ────── Scheduler: with / without dialog ─────── */
const flow   = new Scheduler(psychoJS);
const cancel = new Scheduler(psychoJS);

if (prolificID) {
  /* skip dialog entirely when PID provided */
  psychoJS.schedule(flow);
} else {
  /* show dialog in local / pilot mode */
  psychoJS.schedule(
    psychoJS.gui.DlgFromDict({ dictionary:expInfo, title:'motionlibrary_staircase' })
  );
  psychoJS.scheduleCondition(
    ()=>psychoJS.gui.dialogComponent.button==='OK', flow, cancel
  );
}

/* ────── flow ────── */
flow.add(expInit);
flow.add(instrPracticeBegin); flow.add(instrPracticeLoop); flow.add(instrPracticeEnd);
flow.add(recBegin(true));     flow.add(recLoop(true));     flow.add(recEnd(true));
flow.add(instrMainBegin);     flow.add(instrMainLoop);     flow.add(instrMainEnd);
flow.add(recBegin(false));    flow.add(recLoop(false));    flow.add(recEnd(false));
flow.add(byeBegin);           flow.add(byeLoop);           flow.add(byeEnd);
flow.add(quit,'',true);       cancel.add(quit,'',false);

psychoJS.start({ expName:'motionlibrary_staircase', expInfo });

/* ─────────────────── STIM & STATE ─────────────── */
let win, mouse, square, dot, timerTxt, scoreTxt, choiceTxt;
let phase,snippet,frameSeg,frameTot,control,score,fbUntil;
let vt=[0,0],vd=[0,0],positions,ouVel,trace=[];
let practiceLog=[];
let newTrial=false, waitStart=[0,0], moved=false, isPractice=true;

/* ─────────────────── UTILITIES ──────────────── */
const randn=()=>Math.sqrt(-2*Math.log(Math.random()))*Math.cos(2*Math.PI*Math.random());
function OU(n){const v=Array.from({length:n},()=>[0,0]);for(let t=1;t<n;t++){const[px,py]=v[t-1];v[t]=[px-OU_TH*px+OU_SG*randn(),py-OU_TH*py+OU_SG*randn()];}return v;}
const conf=p=>{const r=Math.hypot(p[0],p[1]);return r<=RADIUS?p:[p[0]*RADIUS/r,p[1]*RADIUS/r];};
const fmt=s=>`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;

/* ─────────────────── INIT EXP ────────────────── */
function expInit(){
  win=psychoJS.window; mouse=new Mouse({win,visible:true});
  square=new visual.Rect({win,width:40,height:40,fillColor:'black',lineColor:'black'});
  const verts=Array.from({length:60},(_,i)=>[Math.cos(2*Math.PI*i/60)*20,Math.sin(2*Math.PI*i/60)*20]);
  dot=new visual.ShapeStim({win,vertices:verts,fillColor:'black',lineColor:'black'});
  const half=win.size[1]/2, UI_X=-(RADIUS+40);
  timerTxt=new visual.TextStim({win,pos:[UI_X,half-40],height:24,color:'white'});
  scoreTxt=new visual.TextStim({win,pos:[UI_X,half-80],height:24,color:'white'});
  choiceTxt=new visual.TextStim({win,pos:[0,half-150],height:32,color:'yellow',alignHoriz:'center',text:'Which one did you control more?'});
  ouVel=OU((PRACTICE_FRAMES+MAIN_FRAMES)*1.2);
  return Scheduler.Event.NEXT;
}

/* ────── INSTRUCTIONS Routines (enhanced with mouse check) ────── */
let instr;
function instrPracticeBegin(){
  instr=new visual.TextStim({win,height:32,color:'white',wrapWidth:1600,alignHoriz:'center',
  text:`Welcome to the Motion Control Study\n\n⚠️ IMPORTANT REQUIREMENT CHECK ⚠️\n\nAre you currently using a computer mouse or a laptop trackpad or touchpad?\n\nPress M if using a MOUSE \nPress T if using a TRACKPAD \n\nNote: You will not receive payment if you cannot complete\nthe study due to not having the required equipment.`});
  return Scheduler.Event.NEXT;
}

function instrPracticeLoop(){
  instr.setAutoDraw(true);
  const keys = psychoJS.eventManager.getKeys({keyList:['m','t']});
  if(keys.length){
    if(keys[0] === 't'){
      // Exit without completion - no redirect to Prolific
      psychoJS.experiment.addData('exit_reason', 'no_mouse_device');
      psychoJS.experiment.addData('device_type', 'trackpad');
      psychoJS.experiment.nextEntry();
      return quit('Study requires mouse device. Exiting without completion.', false);
    } else if(keys[0] === 'm'){
      // Record they have a mouse and continue
      psychoJS.experiment.addData('device_type', 'mouse');
      psychoJS.experiment.nextEntry();
      // Update instruction text for actual task instructions
      instr.text = `Great! Let's begin.\n\nIn this experiment, you will see two shapes on the screen:\na SQUARE and a CIRCLE\n\nYour task is to move your mouse and determine which shape\nyou have more control over.\n\nOne shape will follow your mouse movements more closely,\nwhile the other will move more independently.\n\nAfter each trial, click on the shape you felt you controlled more.\n\nWe'll start with ${PRACTICE_TRIALS} practice trials to help you learn the task.\n\nPress S to begin the practice.`;
      return Scheduler.Event.FLIP_REPEAT; // Stay in loop but with new instructions
    }
  }
  // Check for 'S' press after they've confirmed mouse
  if(psychoJS.experiment._trialsData.length > 0 && psychoJS.eventManager.getKeys({keyList:['s']}).length){
    return Scheduler.Event.NEXT;
  }
  return Scheduler.Event.FLIP_REPEAT;
}

function instrPracticeEnd(){instr.setAutoDraw(false);return Scheduler.Event.NEXT;}

function instrMainBegin(){instr=new visual.TextStim({win,height:32,color:'white',wrapWidth:1600,alignHoriz:'center',text:`Great job on the practice!\n\nNow we'll begin the main experiment with ${MAIN_TRIALS} trials.\n\nPERFORMANCE BONUS:\nYou will earn $0.05 USD for each correct response above 60.\nFor example: 75 correct = 15 × $0.05 = $0.75 bonus\n\nRemember:\n• Move your mouse freely to control the shapes\n• After each movement period, click on the shape you controlled more\n• The shape you controlled will turn GREEN (correct)\n• The other shape will turn RED (incorrect)\n• Try to be as accurate as possible\n\nYour score will be displayed in the top left corner.\n\nPress S to begin the main experiment.`});return Scheduler.Event.NEXT;}
function instrMainLoop(){instr.setAutoDraw(true);if(psychoJS.eventManager.getKeys({keyList:['s']}).length)return Scheduler.Event.NEXT;return Scheduler.Event.FLIP_REPEAT;}
function instrMainEnd(){instr.setAutoDraw(false);return Scheduler.Event.NEXT;}

/* ────── RECORDING BEGIN ────── */
function recBegin(practice){return function(){
  isPractice=practice; phase='WAIT_MOVE';
  snippet=frameSeg=frameTot=0; control='square'; score=0;
  vt=[0,0]; vd=[0,0]; trace=[]; practiceLog=[]; pathLens=[]; minMove=null;
  newTrial=true; moved=false; consecutiveCorrect=consecutiveErrors=0;

  const totalFrames=practice?PRACTICE_FRAMES:MAIN_FRAMES;
  const nSnip=Math.ceil(totalFrames/SNIP_LEN);
  positions=[...Array(nSnip/2).fill(true),...Array(nSnip/2).fill(false)]; util.shuffle(positions);

  square.setAutoDraw(true); dot.setAutoDraw(true);
  timerTxt.setAutoDraw(true); scoreTxt.setAutoDraw(!practice);
  return Scheduler.Event.NEXT;};}

/* ────── RECORDING LOOP ────── */
function recLoop(practice){return function(){
  const [mx,my]=mouse.getPos(), btns=mouse.getPressed();
  const maxF=practice?PRACTICE_FRAMES:MAIN_FRAMES;
  timerTxt.text='Time  '+fmt(Math.floor((maxF-frameTot)/FPS));
  if(!practice) scoreTxt.text='Score '+String(score).padStart(3,'0');

  switch(phase){
    case'WAIT_MOVE':{
      if(newTrial){waitStart=[mx,my];moved=false;newTrial=false;}
      const left=positions[snippet];
      square.setPos(left?[-OFFSET_X,0]:[OFFSET_X,0]);
      dot.setPos   (left?[ OFFSET_X,0]:[-OFFSET_X,0]);
      if(!moved && Math.hypot(mx-waitStart[0],my-waitStart[1])>0){
        moved=true; vt=[0,0]; vd=[0,0]; frameSeg=0; phase='MOVE';
      }
    }break;

    case'MOVE':{
      /* stream frame row */
      psychoJS.experiment.addData('isPractice', practice?1:0);
      psychoJS.experiment.addData('frame',      frameTot);
      psychoJS.experiment.addData('x',          mx);
      psychoJS.experiment.addData('y',          my);
      psychoJS.experiment.nextEntry();

      trace.push([mx,my]);
      const [dmx,dmy]=frameTot?[mx-trace.at(-2)[0], my-trace.at(-2)[1]]:[0,0];
      let [odx,ody]=ouVel[frameTot];
      const mM=Math.hypot(dmx,dmy),oM=Math.hypot(odx,ody); if(oM){odx*=mM/oM;ody*=mM/oM;}
      const tdx=selfProp*dmx+(1-selfProp)*odx, tdy=selfProp*dmy+(1-selfProp)*ody;
      vt[0]=LOWPASS*vt[0]+(1-LOWPASS)*tdx; vt[1]=LOWPASS*vt[1]+(1-LOWPASS)*tdy;
      vd[0]=LOWPASS*vd[0]+(1-LOWPASS)*odx; vd[1]=LOWPASS*vd[1]+(1-LOWPASS)*ody;

      if(control==='square'){
        square.setPos(conf([square.pos[0]+vt[0], square.pos[1]+vt[1]]));
        dot.setPos   (conf([dot.pos[0]   +vd[0], dot.pos[1]   +vd[1]]));
      } else {
        dot.setPos   (conf([dot.pos[0]   +vt[0], dot.pos[1]   +vt[1]]));
        square.setPos(conf([square.pos[0]+vd[0], square.pos[1]+vd[1]]));
      }

      frameSeg++; frameTot++;

      if(frameSeg>=SNIP_LEN||frameTot>=maxF){
        /* path for low-movement flag */
        let path=0;
        for(let f=1;f<SNIP_LEN;f++){
          const [x0,y0]=trace[trace.length-SNIP_LEN-1+f];
          const [x1,y1]=trace[trace.length-SNIP_LEN  +f];
          path+=Math.hypot(x1-x0,y1-y0);
        }
        if(practice){
          pathLens.push(path);
          if(pathLens.length===PRACTICE_TRIALS){
            const med=pathLens.slice().sort((a,b)=>a-b)[Math.floor(pathLens.length/2)];
            minMove=0.70*med;
          }
        } else if(minMove!==null){
          psychoJS.experiment.addData('lowMovement', path<minMove?1:0);
          psychoJS.experiment.nextEntry();
        }
        fbUntil=performance.now()+DRIFT_FREEZE_SEC*1000; phase='FREEZE_DRIFT';
      }
    }break;

    case'FREEZE_DRIFT':{
      if(performance.now()>=fbUntil){
        const left=positions[snippet];
        square.setPos(left?[-OFFSET_X,0]:[OFFSET_X,0]);
        dot.setPos   (left?[ OFFSET_X,0]:[-OFFSET_X,0]);
        choiceTxt.setAutoDraw(true); phase='CHOICE';
      }
    }break;

    case'CHOICE':{
      if(btns[0]){
        const hitSq=Math.abs(mx-square.pos[0])<=20&&Math.abs(my-square.pos[1])<=20;
        const hitDt=Math.hypot(mx-dot.pos[0],my-dot.pos[1])<=20;
        if(hitSq||hitDt){
          const correct=(hitSq?'square':'dot')===control;
          if(practice) practiceLog.push([snippet,selfProp.toFixed(3),correct?1:0]);
          if(!practice&&correct) score++;
          else if(practice){
            if(correct){consecutiveCorrect++;consecutiveErrors=0;if(consecutiveCorrect===2){selfProp=Math.max(MIN_PROP,selfProp-STEP);consecutiveCorrect=0;}}
            else       {consecutiveErrors++;consecutiveCorrect=0;if(consecutiveErrors===1){selfProp=Math.min(MAX_PROP,selfProp+STEP);consecutiveErrors=0;}}
          }
          (control==='square'?square:dot).fillColor='green';
          (control==='square'?dot:square).fillColor='red';
          choiceTxt.setAutoDraw(false);
          fbUntil=performance.now()+FEED_SEC*1000; phase='FEEDBACK';
        }
      }
    }break;

    case'FEEDBACK':{
      if(performance.now()>=fbUntil){
        square.fillColor='black'; dot.fillColor='black';
        control = control==='square'?'dot':'square';
        snippet++; newTrial=true; moved=false;
        phase = frameTot>=maxF?'DONE':'WAIT_MOVE';
      }
    }break;

    case'DONE':
      return Scheduler.Event.NEXT;
  }

  if(psychoJS.eventManager.getKeys({keyList:['escape']}).length) return quit();
  return Scheduler.Event.FLIP_REPEAT;
};}

/* ────── RECORDING END ────── */
function recEnd(practice){return function(){
  square.setAutoDraw(false); dot.setAutoDraw(false);
  timerTxt.setAutoDraw(false); scoreTxt.setAutoDraw(false); choiceTxt.setAutoDraw(false);

  if(practice){
    for(const [t,prop,corr] of practiceLog){
      psychoJS.experiment.addData('practiceTrial', t);
      psychoJS.experiment.addData('propBefore',    prop);
      psychoJS.experiment.addData('practiceCorr',  corr);
      psychoJS.experiment.nextEntry();
    }
    psychoJS.experiment.addData('stair_converged', practiceLog.length>0?1:0);
    psychoJS.experiment.addData('stair_finalProp', selfProp.toFixed(3));
    psychoJS.experiment.addData('personalMinMove', minMove?minMove.toFixed(1):'NA');
    psychoJS.experiment.nextEntry();
  } else {
    // Save final score and bonus calculation for main task
    const correctAbove60 = Math.max(0, score - 60);
    const bonusUSD = (correctAbove60 * 0.05).toFixed(2);
    
    psychoJS.experiment.addData('final_score', score);
    psychoJS.experiment.addData('correct_above_60', correctAbove60);
    psychoJS.experiment.addData('bonus_usd', bonusUSD);
    psychoJS.experiment.addData('participant_id', prolificID);
    psychoJS.experiment.nextEntry();
  }
  return Scheduler.Event.NEXT;};}

/* ────── THANK YOU & REDIRECT ────── */
let bye, byeTimer;
function byeBegin(){
  const correctAbove60 = Math.max(0, score - 60);
  const bonusUSD = (correctAbove60 * 0.05).toFixed(2);
  
  bye=new visual.TextStim({win,height:32,color:'white',text:`Thank you for participating!\n\nYour final score: ${score} out of ${MAIN_TRIALS} correct\n\nExpected bonus: $${bonusUSD} USD\n(for ${correctAbove60} correct responses above 60)\n\nYour responses have been recorded.\nBonus will be paid within 48 hours.\n\nYou will be redirected to Prolific in a moment...`});
  bye.setAutoDraw(true);
  byeTimer=new util.CountdownTimer(4);
  return Scheduler.Event.NEXT;
}
function byeLoop(){return byeTimer.getTime()>0?Scheduler.Event.FLIP_REPEAT:Scheduler.Event.NEXT;}
function byeEnd(){bye.setAutoDraw(false);return Scheduler.Event.NEXT;}

/* ────── QUIT WITH REDIRECT ────── */
function quit(msg='',done=false){
  psychoJS.quit({message:msg,isCompleted:done}).then(()=>{
      // Only redirect if study was completed AND we have a Prolific ID
      if(done && prolificID){
          window.location.href = COMPLETE_URL;
      }
      // If done=false (early exit), no redirect happens
      // Participant returns to Prolific without completion code
  });
  return Scheduler.Event.QUIT;
}
