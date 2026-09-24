'use strict';
(() => {
  const el = id => document.getElementById(id);
  const colors = {grid:'#303438', text:'#89949d', measured:'#87cbb8', target:'#82acf1', inference:'#b3a0df', hold:'#e3ad62', paused:'#76818e'};
  let data = null, status = null, connected = false, telemetryConnected = false;
  let joint = 0, seconds = 30, hover = null, lastGood = 0;
  const finite = Number.isFinite;
  const timeLabel = t => t < 0 ? '' : `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2,'0')}`;
  const value = n => finite(n) ? n.toFixed(3) : '—';
  const fmt = n => finite(n) ? n.toFixed(1) : '—';
  window.armGraphs = {update(s, online) {status=s; connected=online; draw();}};

  function context(id) {
    const canvas=el(id), box=canvas.getBoundingClientRect(), ratio=Math.min(window.devicePixelRatio||1,2);
    const width=Math.round(box.width), height=Math.round(box.height);
    if(canvas.width!==Math.round(width*ratio)||canvas.height!==Math.round(height*ratio)){
      canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
    }
    const c=canvas.getContext('2d');c.setTransform(ratio,0,0,ratio,0,0);c.clearRect(0,0,width,height);
    c.font='10px -apple-system, BlinkMacSystemFont, sans-serif';
    const end=Math.max(1,data?.end||0), start=Math.max(0,end-seconds), span=Math.max(1,end-start);
    const left=52, right=width-10;
    return {c,width,height,left,right,start,end,x:t=>left+(t-start)/span*(right-left)};
  }
  function axes(g, top, bottom) {
    const {c,left,right,start,end,x}=g;
    c.lineWidth=1;c.setLineDash([]);c.textAlign='center';c.fillStyle=colors.text;
    for(let i=0;i<=5;i++){
      const t=start+(end-start)*i/5, px=x(t);
      c.strokeStyle=colors.grid;c.beginPath();c.moveTo(px,top);c.lineTo(px,bottom);c.stroke();
      c.textAlign=i===0?'left':i===5?'right':'center';c.fillText(timeLabel(t),px,bottom+18);
    }
    c.textAlign='left';c.fillText('Time',0,bottom+18);
  }
  function cursor(g, top, bottom){
    if(hover===null || hover<g.start || hover>g.end)return;
    const c=g.c;c.setLineDash([3,3]);c.strokeStyle='#9aa4ad';c.beginPath();c.moveTo(g.x(hover),top);c.lineTo(g.x(hover),bottom);c.stroke();c.setLineDash([]);
  }
  function visible(item,g){return item.t>=g.start && item.t<=g.end;}
  function timeline(){
    const g=context('timeline'),{c,x,left,right,start,end}=g;
    axes(g,6,83);
    c.fillStyle=colors.text;c.textAlign='left';c.fillText('Infer',0,24);c.fillText('Action',0,50);c.fillText('Hold',0,76);
    c.save();c.beginPath();c.rect(left,0,right-left,83);c.clip();
    for(const p of data?.pauses||[]){
      if((p.end??end)<start || p.t>end)continue;
      c.fillStyle='#76818e25';c.fillRect(x(p.t),4,x(p.end??end)-x(p.t),76);
      c.fillStyle=colors.paused;c.fillRect(x(p.t),65,Math.max(2,x(p.end??end)-x(p.t)),12);
    }
    c.fillStyle=colors.inference;
    for(const item of data?.chunks||[]){
      if(item.t<start || item.start>end)continue;
      c.fillRect(x(item.start),13,Math.max(2,x(item.t)-x(item.start)),13);
    }
    c.strokeStyle=colors.measured;c.lineWidth=1;
    c.beginPath();for(const a of data?.actions||[]){if(visible(a,g)){const px=x(a.t);c.moveTo(px,37);c.lineTo(px,53);}}c.stroke();
    c.fillStyle=colors.hold;
    for(const h of data?.holds||[]){if(h.t+(h.duration||0)<start||h.t>end)continue;c.fillRect(x(h.t),65,Math.max(2,x(h.t+(h.duration||0))-x(h.t)),12);}
    c.restore();cursor(g,6,83);
  }
  function trajectory(){
    const g=context('trajectory'), {c,x,left,right,height,start,end}=g;
    const top=12,bottom=height-30;
    const samples=(data?.measured||[]).filter(p=>visible(p,g)&&finite(p.q?.[joint]));
    const actions=data?.actions||[];
    const targets=actions.filter((p,i)=>p.t<=end && (actions[i+1]?.t??end)>=start && finite(p.target?.[joint]));
    const values=[...samples.map(p=>p.q[joint]),...targets.map(p=>p.target[joint])];
    let low=values.length?Math.min(...values):-.1,high=values.length?Math.max(...values):.1;
    const margin=Math.max(.005,(high-low)*.14);low-=margin;high+=margin;
    const y=v=>bottom-(v-low)/(high-low)*(bottom-top);
    axes(g,top,bottom);
    c.textAlign='right';c.fillStyle=colors.text;
    for(let i=0;i<=4;i++){
      const v=low+(high-low)*i/4, py=y(v);c.strokeStyle=colors.grid;c.beginPath();c.moveTo(left,py);c.lineTo(right,py);c.stroke();c.fillText(v.toFixed(3),left-8,py+3);
    }
    c.save();c.beginPath();c.rect(left,top,right-left,bottom-top);c.clip();
    const breaks=[...(data?.chunks||[]).map(p=>p.start),...(data?.holds||[]).map(p=>p.t),...(data?.pauses||[]).map(p=>p.t)].sort((a,b)=>a-b);
    c.strokeStyle=colors.target;c.lineWidth=1.4;c.setLineDash([5,3]);c.beginPath();
    let previousEnd=null,previousValue=null;
    for(let i=0;i<actions.length;i++){
      const a=actions[i];if(a.t>end || !finite(a.target?.[joint]))continue;
      let stop=Math.min(end,actions[i+1]?.t??end);
      const interruption=breaks.find(t=>t>=a.t && t<=stop);if(interruption!==undefined)stop=interruption;
      if(stop<start)continue;
      const from=Math.max(start,a.t),v=a.target[joint];
      if(previousEnd!==null&&Math.abs(previousEnd-a.t)<.000001&&a.t>=start){c.moveTo(x(a.t),y(previousValue));c.lineTo(x(a.t),y(v));}
      c.moveTo(x(from),y(v));c.lineTo(x(stop),y(v));previousEnd=stop;previousValue=v;
    }
    c.stroke();c.setLineDash([]);c.strokeStyle=colors.measured;c.lineWidth=1.8;c.beginPath();
    let previous=null;
    for(const p of samples){if(!previous || p.t-previous.t>.2)c.moveTo(x(p.t),y(p.q[joint]));else c.lineTo(x(p.t),y(p.q[joint]));previous=p;}
    c.stroke();
    if(samples.length&&data?.measured_source==='action_dispatch'){
      c.fillStyle=colors.measured;for(const p of samples){c.beginPath();c.arc(x(p.t),y(p.q[joint]),1.5,0,Math.PI*2);c.fill();}
    }
    c.restore();cursor(g,top,bottom);
    if(hover!==null && hover>=start && hover<=end){
      const p=samples.reduce((best,p)=>!best||Math.abs(p.t-hover)<Math.abs(best.t-hover)?p:best,null);
      if(p&&Math.abs(p.t-hover)<.25){c.fillStyle=colors.measured;c.beginPath();c.arc(x(p.t),y(p.q[joint]),3,0,Math.PI*2);c.fill();}
    }
  }
  function readout(){
    if(!data?.session){el('graph-readout').textContent='Move over a graph to inspect a sample.';return;}
    const t=hover??data.end, a=[...(data.actions||[])].reverse().find(p=>p.t<=t);
    const nearest=(data.measured||[]).reduce((best,p)=>!best||Math.abs(p.t-t)<Math.abs(best.t-t)?p:best,null);
    const measured=nearest&&Math.abs(nearest.t-t)<.25?nearest.q?.[joint]:null;
    const chunk=(data.chunks||[]).find(p=>p.start<=t&&p.t>=t);
    const held=(data.holds||[]).find(p=>Math.abs(p.t-t)<.08);
    let text=`${timeLabel(t)} · J${joint+1} ${value(measured)} rad`;
    if(chunk)text+=` · Inference #${chunk.id+1} · ${fmt(chunk.ms)} ms`;
    else if(held)text+=' · '+(held.joints?.length?'Rejected J'+held.joints.join(', J'):'Validation hold');
    else if(a)text+=` · Chunk #${a.chunk+1}, action ${a.index+1}`;
    el('graph-readout').textContent=text;
  }
  function draw(){
    if(!el('timeline'))return;
    const current=data?.session&&status?.run?.receipt===data.session;
    const runtimeLoading=['loading','warming','maintenance','starting'].includes(status?.phase);
    const online=connected&&telemetryConnected&&performance.now()-lastGood<3000;
    el('graph-status').textContent=!online?'Disconnected · last samples':runtimeLoading?'Previous session':data?.live&&current?(status?.phase==='paused'?'Paused':data.sample_age_s>1?'Waiting for feedback':'Live · 400 ms refresh'):'Last session';
    const latest=data?.measured?.at(-1);
    document.querySelectorAll('[data-joint]').forEach(button=>{
      const index=Number(button.dataset.joint);button.setAttribute('aria-pressed',String(index===joint));button.querySelector('span').textContent=value(latest?.q?.[index]);
    });
    el('graph-source').textContent=data?.measured_source==='action_dispatch'?'Measured at dispatch · session time':'Controller feedback · session time';
    const empty=!data?.actions?.length&&!data?.chunks?.length&&!data?.measured?.length;
    el('graph-empty').hidden=!empty;el('graph-empty').textContent=runtimeLoading?'Waiting for the session to start.':'Start a session to see activity.';
    el('trajectory').setAttribute('aria-label',`Joint ${joint+1} measured and commanded positions in radians. Latest measured ${value(latest?.q?.[joint])}.`);
    timeline();trajectory();readout();
  }
  function initialize(){
    el('graph-window').addEventListener('change',()=>{seconds=Number(el('graph-window').value);hover=null;draw();});
    document.querySelectorAll('[data-joint]').forEach(button=>button.addEventListener('click',()=>{joint=Number(button.dataset.joint);draw();}));
    for(const id of ['timeline','trajectory']){
      el(id).addEventListener('pointermove',event=>{
        const box=el(id).getBoundingClientRect(),end=Math.max(1,data?.end||0),start=Math.max(0,end-seconds);
        const fraction=Math.max(0,Math.min(1,(event.clientX-box.left-52)/(box.width-62)));
        hover=start+fraction*(end-start);draw();
      });
      el(id).addEventListener('pointerleave',()=>{hover=null;draw();});
    }
    new ResizeObserver(draw).observe(el('timeline').parentElement);
    async function poll(){
      try{
        if(!document.hidden){
          const response=await fetch('/api/telemetry',{signal:AbortSignal.timeout(3000)});
          if(!response.ok)throw Error('Telemetry unavailable');
          data=await response.json();telemetryConnected=true;lastGood=performance.now();draw();
        }
      }catch(_){telemetryConnected=false;draw();}
      finally{setTimeout(poll,document.hidden?1500:400);}
    }
    poll();
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',initialize);else initialize();
})();
