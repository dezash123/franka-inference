'use strict';
const $=id=>document.getElementById(id);let state=null,initialized=false,dirty=false,sending=false,connected=false,errorFromServer=false;
const edits=['prompt','actions','backend','action-hz'];
for(const id of edits)$(id).addEventListener('input',()=>{dirty=true;render();});
function error(message){errorFromServer=false;$('error').hidden=!message;$('error').textContent=message||'';}
async function command(path,body={}){if(sending)return;sending=true;render();error('');try{const response=await fetch('/api/'+path,{method:'POST',headers:{'Content-Type':'application/json','X-Control-Token':state.csrf},body:JSON.stringify(body)});const result=await response.json();if(!response.ok)throw Error(result.error);state=result;if(path==='settings')dirty=false;render();}catch(e){error(e.message);}finally{sending=false;render();}}
$('save').onclick=()=>command('settings',{prompt:$('prompt').value,actions:Number($('actions').value),backend:$('backend').value,action_hz:Number($('action-hz').value)});
$('start').onclick=()=>command('start');$('stop').onclick=()=>command('stop');$('pause').onclick=()=>command(['paused','pausing'].includes(state.phase)?'play':'pause');
function renderLatency(s,r){
const candidate=r.latency,unavailable=['loading','warming','maintenance'].includes(s.phase);
const l=!unavailable&&candidate?.backend===s.settings.backend?candidate:null;
const ms=value=>typeof value==='number'&&Number.isFinite(value)?value.toFixed(1):'—';
$('latency-latest').textContent=ms(l?.latest_ms);$('latency-median').textContent=ms(l?.median_ms);$('latency-p95').textContent=ms(l?.p95_ms);
if(!l){$('latency-status').textContent=unavailable?'Runtime loading':'Waiting for inference';$('latency-detail').textContent='No measurements';return;}
const age=Number.isFinite(l.updated_unix)?Math.max(0,Math.floor(s.server_unix-l.updated_unix)):null;
const running=['playing','resuming','pausing','starting'].includes(s.phase);
$('latency-status').textContent=!connected?'Disconnected · last reading':['paused','pausing'].includes(s.phase)?'Paused · last reading':running?(age!==null&&age>5?'Waiting · last reading':'Live'):'Last session';
$('latency-detail').textContent='Round trip '+ms(l.round_trip_ms)+' ms · Last '+l.samples+' inferences'+(age!==null?' · '+age+'s ago':'');
}
function render(){if(!state)return;const s=state,r=s.run||{},motion=['starting','playing','pausing','paused','resuming','stopping'].includes(s.phase);for(const id of edits)$(id).disabled=s.busy||sending||!connected;
$('save').disabled=s.busy||sending||!connected;$('save').textContent=dirty?'Save settings':s.ready?'Save settings':'Load model';
$('start').disabled=s.busy||!s.ready||dirty||sending||!connected||s.robot?.operating_mode!=='Execution';$('pause').disabled=!['playing','paused','pausing','resuming'].includes(s.phase)||sending||!connected;$('pause').textContent=['paused','pausing'].includes(s.phase)?'Play':'Pause';$('stop').disabled=!motion||sending||!connected;
const hz=Number($('action-hz').value),steps=Number($('actions').value);$('timing-hint').textContent=Number.isFinite(hz)&&hz>0&&steps>0?steps+' actions at '+hz+' Hz · '+Math.round(1000*steps/hz)+' ms playback, then fresh inference.':'Set actions per inference and playback Hz.';
$('action-rate').textContent=(r.average_action_hz||0).toFixed(1);$('inference-rate').textContent=(r.average_inference_hz||0).toFixed(1);
renderLatency(s,r);
window.armGraphs?.update(s,connected);
$('model-summary').textContent=s.settings.backend==='ssog_mc2'?'Franka FR3 · π0.5 jointpos · 2 model passes':s.settings.backend==='ov_a_w8a8'?'Franka FR3 · π0.5 jointpos · OpenVINO W8A8 · 5 Euler steps':'Franka FR3 · π0.5 · 5 denoise steps';
$('phase').textContent=s.phase; $('phase-dot').style.background=s.phase==='playing'?'#87cbb8':s.phase==='error'?'#e09c9c':'#88949e';$('message').textContent=s.message;$('saved').textContent=dirty?'Unsaved changes':s.ready?'Model ready'+(s.warmup?.inference_ms?' · '+Math.round(s.warmup.inference_ms.slice(-5).reduce((a,b)=>a+b,0)/5)+' ms warmed':''):'Load the model to start.';
const elapsed=Math.max(0,Math.floor(r.elapsed_s||0));const hours=Math.floor(elapsed/3600);$('elapsed').textContent=(hours?String(hours).padStart(2,'0')+':':'')+String(Math.floor(elapsed/60)%60).padStart(2,'0')+':'+String(elapsed%60).padStart(2,'0');$('action-count').textContent=r.actions||0;$('inference-count').textContent=r.inferences||0;$('hold-count').textContent=((r.joint_holds||0)+(r.workspace_holds||0)+(r.validation_holds||0))+' / '+(r.camera_holds||0);
const robot=s.robot;const brief=robot?[robot.operating_mode||robot.error,(robot.brakes||[]).every(b=>b==='Unlocked')?'Brakes unlocked':'Brakes locked',...(robot.robot_errors||[])].filter(Boolean).join(' · '):'Checking…';$('robot').textContent=brief.length>180?brief.slice(0,180)+'…':brief;
if(s.phase==='error'){error(s.message);errorFromServer=true;}else if(errorFromServer){error('');}
}
async function poll(){try{const response=await fetch('/api/status');if(!response.ok)throw Error('Connection lost');state=await response.json();connected=true;if(!initialized){$('prompt').value=state.settings.prompt;$('actions').value=state.settings.actions;$('backend').value=state.settings.backend;$('action-hz').value=state.settings.action_hz;initialized=true;}else if(!dirty&&!sending){$('prompt').value=state.settings.prompt;$('actions').value=state.settings.actions;$('backend').value=state.settings.backend;$('action-hz').value=state.settings.action_hz;}$('connected').textContent='Connected';$('connection').className='online';render();}catch(e){connected=false;$('connected').textContent='Disconnected';$('connection').className='';render();}finally{setTimeout(poll,700);}}
const cameraCards=new Map();
function cameraCard(camera,index){
let card=cameraCards.get(camera.id);if(card)return card;
const article=document.createElement('article');article.className='panel camera-card';article.dataset.camera=camera.id;
article.innerHTML='<div class="section-title"><h2></h2><span class="camera-fps">— FPS</span></div><div class="image-wrap"><img><span class="image-placeholder">Waiting for camera</span></div><div class="camera-meta"><span class="camera-label"></span><span class="camera-state">Connecting</span></div>';
const img=article.querySelector('img');img.alt=camera.role==='wrist'?'Wrist left-eye camera preview':'External '+index+' left-eye camera preview';
article.querySelector('h2').textContent=camera.role==='wrist'?'Wrist':'External '+index;
article.querySelector('.camera-label').textContent=camera.serial+' · left';
card={article,img,lastFrameRequest:0,available:false};
img.onload=()=>img.classList.toggle('loaded',card.available);
img.onerror=()=>{img.classList.remove('loaded');article.querySelector('.image-placeholder').textContent='Preview unavailable';};
$('cameras').append(article);cameraCards.set(camera.id,card);return card;
}
function renderCameras(data){
const present=new Set();let external=0;
for(const camera of data.cameras){
if(camera.role==='external')external++;
const card=cameraCard(camera,external),article=card.article;present.add(camera.id);
card.available=!!camera.frame_available;
article.dataset.healthy=String(camera.healthy);
article.querySelector('.camera-fps').textContent=Number.isFinite(camera.fps)?camera.fps.toFixed(1)+' FPS':'— FPS';
article.querySelector('.camera-state').textContent=camera.healthy?(camera.inference_enabled?'Live':'Live · preview only'):camera.status;
article.querySelector('.camera-state').title=camera.status;
article.querySelector('.image-placeholder').textContent=camera.status||'Waiting for camera';
if(!card.available)card.img.classList.remove('loaded');
else if(!document.hidden&&Date.now()-card.lastFrameRequest>=900){card.lastFrameRequest=Date.now();card.img.src='/api/frame?camera='+encodeURIComponent(camera.id)+'&t='+card.lastFrameRequest;}
}
for(const [id,card] of cameraCards)if(!present.has(id)){card.article.remove();cameraCards.delete(id);}
}
async function pollCameras(){
try{const response=await fetch('/api/cameras');if(!response.ok)throw Error('Camera status unavailable');renderCameras(await response.json());}
catch(e){for(const card of cameraCards.values()){card.available=false;card.img.classList.remove('loaded');card.article.dataset.healthy='false';card.article.querySelector('.camera-fps').textContent='— FPS';card.article.querySelector('.camera-state').textContent='Status unavailable';card.article.querySelector('.image-placeholder').textContent='Camera status unavailable';}}
finally{setTimeout(pollCameras,document.hidden?2500:1000);}
}
poll();pollCameras();
