'use strict';
const el=id=>document.getElementById(id), status=t=>el('status').textContent=t;
let info, busy=false, sid='', seq=0, stream, ctx,node,source;
let vocabulary={words:[],extras:[]}, vocabularyRenderKey='';
let tail=new Int16Array(0), chain=Promise.resolve(), total=0, recording=false, fail=null;
async function req(path,body={},headers={}){
 const binary=body instanceof Int16Array;
 const r=await fetch(path,{method:'POST',headers:{'Content-Type':binary?'application/octet-stream':'application/json',...headers},body:binary?body:JSON.stringify(body)});
 if(!r.ok){let err;try{err=(await r.json()).error}catch{}throw Error(err||('HTTP '+r.status))}
 return path==='/clip'?new Int16Array(await r.arrayBuffer()):r.json();
}
function controls(){for(const id of ['mic','add','samples','word'])el(id).disabled=busy||!info||info.occupied;el('play').disabled=busy||!info||info.occupied||!info.replayable;el('stop').disabled=!recording;el('mic').disabled=busy||!info||info.occupied||info.selected>=0;document.querySelectorAll('#extras button').forEach(b=>b.disabled=busy||info?.occupied);}
async function sync(){info=await req('/state');vocabulary=await req('/vocabulary');drawVocabulary();el('shared').textContent=info.occupied?'有人正在录音或识别，请稍候再操作。':'共享操作台 · 当前空闲 · 热词与历史共用，请轮流录音';el('samples').replaceChildren(new Option('自由麦克风',-1),...info.clips.map((c,i)=>new Option(`实测样例 ${i+1} · ${c.word} · ${c.source} · ${c.seconds.toFixed(1)} 秒`,i)));el('samples').value=info.selected;const c=info.clips[info.selected];el('hint').textContent=c?`本段目标：${c.word} · ${Object.hasOwn(info.extras,c.word)?'已加入本场词库':'尚未加入本场词库'}`:'输入中文人名、地名或品牌，下一段音频生效。';el('online').textContent=`GPU ${info.gpu} 已连接 · ${info.size} 个热词`;draw();controls();}
let micFocus='';
function tag(parent,name,text,cls=''){const n=document.createElement(name);n.textContent=text;if(cls)n.className=cls;parent.append(n);return n}
function draw(){
 const hist=info?.history||[], extras=info?.extras||{};
 const list=el('extras');list.replaceChildren();
 for(const [w,phones] of Object.entries(extras)){
   const row=tag(list,'div','','hot');tag(row,'b',w);tag(row,'small',phones);
   tag(row,'span',info.overrides.includes(w)?'已生效 · 覆盖基础读音':'已生效','feedback');
   const btn=tag(row,'button','移除');btn.disabled=busy||info?.occupied;btn.onclick=()=>action(async()=>{
     const res=await req('/remove',{word:w});if(info.selected<0)micFocus=w;await sync();
     status(res.background?'已移除临时配置 · 已恢复背景词库读音':'已移除 · 可重播同一句');});
 }
 if(!list.childNodes.length)tag(list,'small','暂无临时热词');
 const out=el('history');out.replaceChildren();
 if(!hist.length)tag(out,'small','尚无记录。添加热词或完成一次转写后会出现在这里。');
 for(const e of [...hist].reverse()){
   const row=tag(out,'article','','event');
   const title=e.kind==='test'?'转写定稿':(e.kind==='add'?'添加热词：':'移除热词：')+e.word;
   tag(row,'b',title);tag(row,'small','　'+new Date(e.time).toLocaleTimeString()+' · '+[e.source,e.target,e.sample].filter(Boolean).join(' / '));
   if(e.kind==='test'){
     const cols=tag(row,'div','','versions');
     for(const [label,text] of [['基础模型',e.left],['热词增强',e.right]]){
       const cell=tag(cols,'div','');tag(cell,'small',label);tag(cell,'p',text||'（空转写）');
     }
     tag(row,'small','本次各步实际提示：'+(e.prompted.join('、')||'无'));
   }
   const words=Object.keys(e.table.extras);
   const detail=tag(row,'details','');
   tag(detail,'summary','当时生效 '+e.table.size+' 词 · 本场配置：'+(words.join('、')||'无'));
   tag(detail,'p','基础词表版本 '+e.table.base.slice(0,12)+(e.table.omitted?'；背景索引排除本例原词：'+e.table.omitted:'；使用完整背景词表'));
   for(const [w,ph] of Object.entries(e.table.extras))tag(detail,'p',w+' → '+ph);
 }

}
el('word').oninput=()=>{if(info&&info.selected<0){micFocus=el('word').value.trim();draw()}};
function render(a){const focus=el('word').value.trim();for(const [k,rev] of [['left','lr'],['right','rr']]){const d=el(k),prev=d.textContent,t=a[k];let i=0;while(i<prev.length&&i<t.length&&prev[i]===t[i])i++;if(i<prev.length)el(rev).textContent='刚刚回改：'+prev.slice(i)+' → '+(t.slice(i)||'（删除）');d.replaceChildren();if(focus&&t.includes(focus)){t.split(focus).forEach((s,i)=>{if(i){let m=document.createElement('mark');m.textContent=focus;d.append(m)}d.append(s)})}else d.textContent=t;}
 const h=el('hits');h.replaceChildren();for(const c of a.candidates){const span=document.createElement('span');span.className='chip';span.textContent=`${c.word} · ${c.score.toFixed(2)}`;h.append(span)}if(!a.candidates.length)h.textContent='本步没有候选通过';status(`${a.final?'已定稿':'识别中'} · ${a.seconds.toFixed(1)} 秒`);}
async function chunk(p,final){const a=await req('/step',p,{'X-Session':sid,'X-Seq':String(seq++),'X-Final':final?'1':'0'});render(a);if(final){sid='';await sync()}}
async function begin(kind){sid=(await req('/start',{kind})).session;seq=0;total=0;tail=new Int16Array(0);chain=Promise.resolve();fail=null;for(const id of ['left','right','lr','rr','hits'])el(id).replaceChildren();status('正在接收语音…');}
function release(){recording=false;if(node)node.port.onmessage=null;node?.disconnect();source?.disconnect();stream?.getTracks().forEach(t=>t.stop());if(ctx)ctx.close().catch(()=>{});ctx=node=source=stream=null;controls();}
async function abort(){release();if(sid)try{await req('/abort',{}, {'X-Session':sid})}catch{}sid='';}
async function action(fn){if(busy)return;busy=true;controls();try{await fn()}catch(e){await abort();status('未完成：'+e.message)}finally{busy=false;controls()}}
el('samples').onchange=()=>action(async()=>{await req('/select',{i:+el('samples').value});await sync();el('word').value=info.clips[info.selected]?.word||micFocus;['phones','left','right','lr','rr','hits'].forEach(k=>el(k).textContent='');status('已切换音频 · 本场热词继续生效')});
el('add').onclick=()=>action(async()=>{status('正在转换发音并更新检索索引…');const a=await req('/add',{word:el('word').value});if(info.selected<0)micFocus=a.word;el('phones').textContent=`${a.word} → ${a.phones}`;await sync();status('已入库 · 下一段语音生效')});
el('play').onclick=()=>action(async()=>{ctx=new AudioContext({sampleRate:16000});try{await ctx.resume();const pcm=await req('/clip');await begin('replay');const b=ctx.createBuffer(1,pcm.length,16000), f=b.getChannelData(0);pcm.forEach((v,i)=>f[i]=v/32768);source=ctx.createBufferSource();source.buffer=b;source.connect(ctx.destination);const start=ctx.currentTime;source.start();for(let off=0;off<pcm.length;off+=32000){let end=Math.min(off+32000,pcm.length);await new Promise(r=>setTimeout(r,Math.max(0,(start+end/16000-ctx.currentTime)*1000)));await chunk(pcm.slice(off,end),end===pcm.length)}}finally{release()}});
let stopRecording=()=>{};
el('stop').onclick=()=>stopRecording();
el('mic').onclick=()=>action(async()=>{if(!navigator.mediaDevices?.getUserMedia)throw Error('请使用已信任证书的 HTTPS 地址访问，并允许麦克风权限');try{ctx=new AudioContext({sampleRate:16000});await ctx.resume();if(ctx.sampleRate!==16000)throw Error('请使用支持 16 kHz 采集的 Chrome 或 Edge');stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1}});await ctx.audioWorklet.addModule('/mic.js');source=ctx.createMediaStreamSource(stream);node=new AudioWorkletNode(ctx,'pcm-input');const mute=ctx.createGain();mute.gain.value=0;node.connect(mute).connect(ctx.destination);await begin('mic');recording=true;controls();await new Promise(resolve=>{stopRecording=()=>{if(recording){recording=false;controls();resolve()}};node.port.onmessage=({data})=>{if(!recording)return;const n=Math.min(data.length,480000-total),next=new Int16Array(tail.length+n);next.set(tail);for(let i=0;i<n;i++)next[tail.length+i]=Math.max(-32768,Math.min(32767,Math.round(data[i]*32768)));total+=n;tail=next;if(tail.length>=32000){const p=tail.slice(0,32000);tail=tail.slice(32000);chain=chain.then(()=>{if(!fail)return chunk(p,false)}).catch(e=>{fail=e;stopRecording()})}if(total>=480000)stopRecording()};source.connect(node)});release();await chain;if(fail)throw fail;if(!total)throw Error('没有收到音频');await chunk(tail,true)}finally{release()}});
function drawVocabulary(){
 const q=el('word-query').value.trim(), normalized=q.toLocaleLowerCase();
 const renderKey=JSON.stringify([q,vocabulary]);if(renderKey===vocabularyRenderKey)return;vocabularyRenderKey=renderKey;
 const matches=vocabulary.words.filter(w=>w.toLocaleLowerCase().includes(normalized));
 const exact=vocabulary.words.some(w=>w.toLocaleLowerCase()===normalized);
 el('word-count').textContent=`${vocabulary.words.length} 词`;
 el('query-status').textContent=q?`${exact?'已存在该热词':'未找到完全相同的热词'} · ${matches.length} 个包含匹配`:`共 ${matches.length} 词 · 上下滚动查看，拖动底边可调整高度`;
 const list=el('word-list');list.replaceChildren();
 for(const w of matches){const row=tag(list,'div','','word-row');tag(row,'span',w);if(vocabulary.extras.includes(w))tag(row,'small','本场配置')}
 if(!matches.length)tag(list,'small','没有匹配的热词');
}
el('word-query').oninput=drawVocabulary;
async function connect(){try{await sync();el('word').value=info.clips[info.selected]?.word||micFocus;status('已就绪 · 可开始录音')}catch(e){status(e.message)}}
connect();
let refreshing=false;
setInterval(async()=>{if(busy||refreshing||document.hidden)return;refreshing=true;try{await sync()}catch(e){status(e.message)}finally{refreshing=false}},5000);
