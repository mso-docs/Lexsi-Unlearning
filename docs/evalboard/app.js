const $ = s => document.querySelector(s);
const escape = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels = {metrics:'Metrics',overview:'Overview',scoreboard:'Scoreboard',traces:'Trace explorer',improvements:'Improvement tracking',tools:'Tool activity',memory:'Memory events',interp:'Interpretability'};
const icons = {metrics:'▦',overview:'◈',scoreboard:'▥',traces:'≋',improvements:'↗',tools:'⌘',memory:'◇',interp:'⬡'};
let data, active='overview', layoutDraft, selectedTrace, sectionContext=null, selectedTraceVisuals=true;
const activeTabs=Object.create(null);
const staticDataBase=(typeof document!=='undefined'&&document.documentElement&&typeof document.documentElement.getAttribute==='function'&&document.documentElement.getAttribute('data-static-data'))||'';
function staticBase(){return staticDataBase.endsWith('/')||staticDataBase===''?staticDataBase:staticDataBase+'/';}
function staticUrl(path){
 const u=new URL(path,'https://evalboard.invalid');
 if(u.pathname==='/api/workspace')return staticBase()+'workspace.json';
 if(u.pathname==='/api/interp/artifact')return staticBase()+'interp/'+encodeURIComponent(u.searchParams.get('id')||'')+'.json';
 if(u.pathname==='/api/compare')return {compare:true,params:Object.fromEntries(u.searchParams)};
 throw Error('This static snapshot has no endpoint for '+path);
}
function compareFromWorkspace(ws,baseline,candidate,metric){
 const latest={};
 for(const s of ws.scores||[])if(s.metric===metric)latest[s.trace_id]=s.value;
 const runs={};
 for(const t of ws.traces||[]){
  if(t.run_id===baseline||t.run_id===candidate){
   const cases=runs[t.run_id]||(runs[t.run_id]={});
   if(t.case_id in cases)throw Error('Comparison requires one trace per case per run; separate repeated trials into runs');
   cases[t.case_id]=latest[t.id];
  }
 }
 const a=runs[baseline]||{}, b=runs[candidate]||{};
 const paired=Object.keys(a).filter(k=>k in b&&a[k]!=null&&b[k]!=null).sort().map(k=>({case_id:k,baseline:a[k],candidate:b[k],delta:b[k]-a[k]}));
 const union=new Set([...Object.keys(a),...Object.keys(b)]);
 return {metric,baseline,candidate,paired,paired_count:paired.length,excluded_count:union.size-paired.length,mean_delta:paired.length?paired.reduce((s,p)=>s+p.delta,0)/paired.length:null};
}
async function api(path, body) {
 if(staticDataBase){
  if(body!==undefined)throw Error('This is a static snapshot. Scoring, layout edits, and imports are disabled.');
  const mapped=staticUrl(path);
  if(mapped&&mapped.compare){
   const ws=data||await(await fetch(staticBase()+'workspace.json')).then(r=>{if(!r.ok)throw Error('Could not load workspace.json');return r.json();});
   return compareFromWorkspace(ws,mapped.params.baseline,mapped.params.candidate,mapped.params.metric);
  }
  const r=await fetch(mapped);
  const result=await r.json();
  if(!r.ok)throw Error(result.error||'Request failed');
  return result;
 }
 const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const result=await r.json();
 if(!r.ok)throw Error(result.error||'Request failed');
 return result;
}
function notice(message,error=false){$('#notice').textContent=message;$('#notice').className=error?'error':'';}
function latest(trace,metric){return data.scores.filter(s=>s.trace_id===trace.id&&s.metric===metric).at(-1);}
function catalog(){const visible=(data.metric_catalog||[]).filter(m=>m.visible!==false);return sectionContext?.metrics?sectionContext.metrics.map(id=>visible.find(m=>m.id===id)).filter(Boolean):visible;}
function metrics(){return data.metric_catalog?[...new Set(catalog().filter(m=>m.source==='score').map(m=>m.key))]:[...new Set(data.scores.map(s=>s.metric))];}
function metricLabel(key){return catalog().find(m=>m.source==='score'&&m.key===key)?.label||key;}

// Run IDs from harnesses share long prefixes (project, timestamp). Show the part that differs; the full ID stays in tooltips.
let runLabelMap=new Map(),interpLabelMap=new Map();
function computeRunLabels(runs){
 runs=[...new Set(runs.filter(Boolean))];const map=new Map(runs.map(r=>[r,r]));if(runs.length<2)return map;
 const split=r=>r.match(/[^-_:\/]+[-_:\/]?|[-_:\/]/g)||[r];
 const parts=runs.map(split);let pre=0;
 while(parts.every(p=>p.length>pre+1&&p[pre]===parts[0][pre]))pre++;
 let suf=0;
 while(parts.every(p=>p.length>pre+suf+1&&p[p.length-1-suf]===parts[0][parts[0].length-1-suf]))suf++;
 let labels=parts.map(p=>p.slice(pre,p.length-suf).join('').replace(/[-_:\/]+$/,''));
 const unique=ls=>new Set(ls).size===ls.length&&ls.every(Boolean);
 if(!unique(labels))labels=runs.slice();
 const noStamp=labels.map(l=>l.replace(/\d{8}T\d{6}Z[-_:]?/g,'').replace(/^[-_:]+/,''));
 if(unique(noStamp))labels=noStamp;
 runs.forEach((r,i)=>map.set(r,labels[i]));return map;
}
function runLabel(run){return runLabelMap.get(run)||run;}
function interpRunLabel(run){return interpLabelMap.get(run)||run;}
function midEllipsis(s,n){s=String(s??'');if(s.length<=n)return s;const head=Math.ceil((n-1)*.55),tail=n-1-head;return s.slice(0,head)+'…'+s.slice(s.length-tail);}
function runCell(run){return `<span title="${escape(run)}">${escape(runLabel(run))}</span>`;}
function shortTraceId(t){const p=t.run_id+':';return t.id.startsWith(p)?t.id.slice(p.length):t.id;}
function traces(){const run=$('#run-filter').value, q=$('#search').value.toLowerCase();return data.traces.filter(t=>(!run||t.run_id===run)&&(!q||[t.id,t.case_id,t.model,t.run_id].join(' ').toLowerCase().includes(q)));}
function average(values){return values.length?values.reduce((a,b)=>a+b,0)/values.length:null;}
function pct(n){return n===null?'Unscored':`${(n*100).toFixed(1)}%`;}
function table(headers,rows){return `<div class="table-wrap"><table><thead><tr>${headers.map(h=>`<th>${escape(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;}
function panel(title,sub,content){return `<article class="panel"><div class="panel-head"><div><h2>${escape(title)}</h2><p class="muted">${escape(sub)}</p></div></div>${content}</article>`;}
function empty(text){return `<div class="empty">${escape(text)}</div>`;}
function runScores(ts,metric){return [...new Set(ts.map(t=>t.run_id))].map(run=>{const items=ts.filter(t=>t.run_id===run);const values=items.map(t=>latest(t,metric)?.value).filter(v=>v!==undefined);return {run,value:average(values),coverage:`${values.length}/${items.length}`};});}
function traceTable(ts){return ts.length?table(['Case / trace','Run','Model','Status','Duration','Inspect'],ts.map(t=>`<tr><td><strong>${escape(t.case_id)}</strong><br><small class="muted" title="${escape(t.id)}">${escape(midEllipsis(shortTraceId(t),48))}</small></td><td>${runCell(t.run_id)}</td><td>${escape(t.model)}</td><td><span class="badge ${t.status==='error'?'error':''}">${escape(t.status)}</span></td><td>${t.adapter?.missing?.includes('duration_ms')?'Not recorded':t.duration_ms.toLocaleString()+' ms'}</td><td><button class="nowrap" data-trace="${escape(t.id)}">Open trace ↗</button></td></tr>`)):empty('No traces in this scope. Import a JSON or JSONL trace file to begin.');}
// All visualizations use recorded values and include text values and SVG titles.
function visuals(){return data.layout.visuals !== false && sectionContext?.visuals!==false;}
// Default metric: the one marked primary in the metric catalog, else the score metric covering the most traces.
function defaultMetric(){
 const ms=metrics(),flagged=catalog().find(m=>m.primary&&m.source==='score'&&ms.includes(m.key));if(flagged)return flagged.key;
 let best=ms[0],bestN=-1;ms.forEach(k=>{const n=new Set((data.scores||[]).filter(s=>s.metric===k).map(s=>s.trace_id)).size;if(n>bestN){best=k;bestN=n;}});return best;
}
function selectedMetric(){const selected=$('#metric-filter').value;return metrics().includes(selected)?selected:defaultMetric();}
// Horizontal bars are HTML rows with a percent-width SVG bar, so labels keep a readable size at any panel width.
function barChart(items, {label='Results', max=null, format=v=>String(v)}={}) {
 if(!items.length)return empty('No data available in this scope.');
 const limit=max??Math.max(...items.map(item=>item.value),1e-9);
 return `<figure class="chart bar-chart" role="img" aria-label="${escape(label)}"><div class="bar-rows">${items.map(item=>{const w=Math.max(0,Math.min(1,item.value/limit))*100;return `<div class="bar-row" title="${escape(item.full||item.label)}: ${escape(format(item.value))}${item.note?' · '+escape(item.note):''}"><span class="bar-label">${escape(midEllipsis(item.label,64))}</span><svg class="bar-svg" viewBox="0 0 100 20" preserveAspectRatio="none" aria-hidden="true"><rect class="chart-track" x="0" y="0" width="100" height="20"/><rect class="chart-bar ${item.tone||''}" x="0" y="0" width="${w}" height="20"/></svg><span class="bar-value">${escape(format(item.value))}</span></div>`;}).join('')}</div><figcaption>${escape(label)}</figcaption></figure>`;
}
function distribution(ts,metric){
 const bins=[0,0,0,0,0];let count=0;
 ts.forEach(t=>{const score=latest(t,metric);if(score){bins[Math.min(4,Math.floor(score.value*5))]++;count++;}});
 return count?barChart(bins.map((value,i)=>({label:`${i*20}–${(i+1)*20}%${i===4?' (incl. 100%)':''}`,value})),{label:`Score distribution · ${count}/${ts.length} scored · latest ${metricLabel(metric)}`}):empty('No scored traces in this scope. Add scores to see their distribution.');
}
function overviewVisuals(ts,metric){
 const runs=runScores(ts,metric),scored=runs.filter(r=>r.value!==null);
 const performance=barChart(scored.map(r=>({label:runLabel(r.run),full:r.run,value:r.value,note:r.coverage+' scored'})),{label:`Run means · ${metricLabel(metric)||'no metric'} · unpaired descriptive scores`,max:1,format:pct});
 const costs=[...new Set(ts.map(t=>t.run_id))].map(run=>({label:run,values:ts.filter(t=>t.run_id===run).map(t=>traceMetricValue(t,'cost_usd')).filter(v=>v!==null)})).filter(r=>r.values.length).map(r=>({label:runLabel(r.label),full:r.label,value:r.values.reduce((a,b)=>a+b,0)}));
 const durations=[...new Set(ts.map(t=>t.run_id))].map(run=>({label:run,value:average(ts.filter(t=>t.run_id===run).map(t=>traceMetricValue(t,'duration_ms')).filter(v=>v!==null))})).filter(r=>r.value!==null).map(r=>({...r,full:r.label,label:runLabel(r.label)}));
 return `<div class="visual-grid">${panel('Run score comparison','Latest score per trace; coverage is available below.',performance)}${panel('Score distribution','See the spread behind the mean.',distribution(ts,metric))}${panel('Trace outcomes','Execution status is separate from evaluation score.',barChart([{label:'OK',value:ts.filter(t=>t.status==='ok').length},{label:'Error',value:ts.filter(t=>t.status==='error').length,tone:'negative'},{label:'Unknown',value:ts.filter(t=>t.status==='unknown').length}],{label:'Trace status counts'}))}${costs.some(c=>c.value>0)?panel('Cost by run','Total reported cost; run sizes may differ.',barChart(costs,{label:'Reported run cost (USD)',format:v=>'$'+v.toFixed(4)})):''}${panel('Latency by run','Arithmetic mean of reported trace durations.',barChart(durations,{label:'Mean trace duration (milliseconds)',format:v=>v.toFixed(0)+' ms'}))}</div>`;
}
function pairedChart(result){
 if(!visuals()||!result.paired.length)return '';
 const LIM=40,rows=[...result.paired].sort((p,q)=>Math.abs(q.delta)-Math.abs(p.delta)||String(p.case_id).localeCompare(String(q.case_id))),shown=rows.slice(0,LIM);
 const ROW=30,TOP=46,AX=250,AW=300,W=720,H=TOP+shown.length*ROW+10;
 const ticks=[0,.5,1].map(f=>`<g><line class="grid-line" x1="${AX+f*AW}" x2="${AX+f*AW}" y1="${TOP-14}" y2="${H-10}"/><text class="chart-value" x="${AX+f*AW}" y="${TOP-22}" text-anchor="middle">${f*100}%</text></g>`).join('');
 const body=shown.map((p,i)=>{const y=TOP+i*ROW+ROW/2-4,a=AX+p.baseline*AW,b=AX+p.candidate*AW;return `<g><title>${escape(p.case_id)}: ${pct(p.baseline)} to ${pct(p.candidate)}; ${(p.delta*100).toFixed(1)} percentage points</title><text class="chart-label" x="0" y="${y+4}">${escape(midEllipsis(p.case_id,34))}</text><line class="chart-axis" x1="${AX}" x2="${AX+AW}" y1="${y}" y2="${y}"/><line class="chart-connector" x1="${a}" x2="${b}" y1="${y}" y2="${y}"/><circle class="baseline-point" cx="${a}" cy="${y}" r="5"/><circle class="candidate-point" cx="${b}" cy="${y}" r="5"/><text class="chart-value" x="${AX+AW+18}" y="${y+4}">${p.delta>=0?'+':''}${(p.delta*100).toFixed(1)} pp</text></g>`;}).join('');
 const cap=rows.length>LIM?`Showing the ${LIM} of ${rows.length} paired cases with the largest changes; the table below lists all. `:'';
 return `<figure class="chart paired-chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Baseline and candidate scores by matching case"><title>Paired scores: hollow circles are baseline, filled circles are candidate. Axis is zero to one.</title>${ticks}${body}</svg><figcaption>${cap}○ Baseline · ● Candidate · paired cases only · sorted by size of change · descriptive changes</figcaption></figure>`;
}
function traceMetricValue(t,key){
 if(t.adapter?.missing?.includes(key))return null;
 if(key==='execution_ok')return ['ok','error'].includes(t.status)?(t.status==='ok'?1:0):null;
 if(key==='tool_calls')return t.spans.filter(s=>s.kind==='tool').length;
 if(key==='memory_events')return t.memory_events.length;
 return typeof t[key]==='number'&&Number.isFinite(t[key])?t[key]:null;
}
function formatMetric(value,format){if(value===null)return 'No data';return format==='percent'?pct(value):format==='usd'?'$'+value.toFixed(4):format==='ms'?value.toFixed(1)+' ms':Number(value.toFixed(3)).toLocaleString();}
function metricValues(ts,m){return ts.map(t=>m.source==='score'?latest(t,m.key)?.value:traceMetricValue(t,m.key)).filter(v=>v!==undefined&&v!==null);}
function aggregate(values,aggregation){if(!values.length)return null;switch(aggregation){case 'sum':return values.reduce((a,b)=>a+b,0);case 'min':return Math.min(...values);case 'max':return Math.max(...values);case 'count':return values.length;default:return average(values);}}
function metricView(ts){
 const definitions=catalog();if(!definitions.length)return empty('No visible metric definitions. Import traces or customize the metric catalog.');
 return `<div class="visual-grid">${definitions.map(m=>{const values=metricValues(ts,m),value=aggregate(values,m.aggregation),groups=[...new Set(ts.map(t=>t.run_id))].map(run=>{const cases=ts.filter(t=>t.run_id===run),values=metricValues(cases,m);return {label:run,value:aggregate(values,m.aggregation),coverage:values.length+'/'+cases.length};});const format=v=>formatMetric(v,m.aggregation==='count'?'number':m.format);return panel(m.label,`${m.source==='score'?'Attributed evaluation score':'Recorded/derived execution metric'} · ${m.aggregation} · ${m.key}`,`<div class="panel-body"><div class="value metric-total">${escape(format(value))}</div><p class="muted">${values.length}/${ts.length} traces with evidence</p></div>${visuals()&&groups.some(g=>g.value!==null&&g.value>=0)?barChart(groups.filter(g=>g.value!==null&&g.value>=0).map(g=>({...g,full:g.label,label:runLabel(g.label)})),{label:m.label+' by run · nonnegative values',format}):''}${table(['Run',m.label,'Coverage'],groups.map(g=>`<tr><td>${runCell(g.label)}</td><td>${escape(format(g.value))}</td><td>${g.coverage}</td></tr>`))}`);}).join('')}</div>`;
}
// Improvement tracking: preselect the pair of runs that share the most scored cases (ties: most similar run IDs, then latest).
let compareState=null;const compareCache=Object.create(null);
function commonPrefix(a,b){let n=0;while(n<a.length&&n<b.length&&a[n]===b[n])n++;return n;}
function pickComparePair(runs,metric){
 const scoredIds=new Set((data.scores||[]).filter(s=>s.metric===metric).map(s=>s.trace_id)),cases=new Map();
 data.traces.forEach(t=>{if(scoredIds.has(t.id)){if(!cases.has(t.run_id))cases.set(t.run_id,new Set());cases.get(t.run_id).add(t.case_id);}});
 const rs=runs.filter(r=>cases.has(r));let best=null;
 for(let i=0;i<rs.length;i++)for(let j=i+1;j<rs.length;j++){let overlap=0;cases.get(rs[i]).forEach(c=>{if(cases.get(rs[j]).has(c))overlap++;});const key=[overlap,commonPrefix(rs[i],rs[j]),j];if(overlap&&(!best||cmpKey(key,best.key)>0))best={key,pair:[rs[i],rs[j]]};}
 return best?best.pair:[runs[0],runs.at(-1)];
}
function cmpKey(a,b){for(let i=0;i<a.length;i++)if(a[i]!==b[i])return a[i]-b[i];return 0;}
function compareSelection(runs,ms){
 const metric=compareState&&ms.includes(compareState.metric)?compareState.metric:selectedMetric();
 if(compareState&&runs.includes(compareState.baseline)&&runs.includes(compareState.candidate)&&compareState.metric===metric)return compareState;
 const [baseline,candidate]=pickComparePair(runs,metric);compareState={baseline,candidate,metric};return compareState;
}
function compareHtml(result,visualsOn){
 return `<div class="panel-body"><div class="callout"><strong>${result.mean_delta===null?'No paired scores':`${result.mean_delta>=0?'+':''}${(result.mean_delta*100).toFixed(1)} percentage points`}</strong> · ${result.paired_count} paired cases · ${result.excluded_count} excluded. Higher is better for this normalized metric; invert lower-is-better metrics at ingestion.</div></div>`+(visualsOn?pairedChart(result):'')+table(['Case','Baseline','Candidate','Change'],result.paired.map(p=>`<tr><td>${escape(p.case_id)}</td><td>${pct(p.baseline)}</td><td>${pct(p.candidate)}</td><td class="${p.delta>=0?'positive':'negative'}">${p.delta>=0?'+':''}${(p.delta*100).toFixed(1)} pp</td></tr>`));
}
async function runCompare(form){
 const out=form?.nextElementSibling;if(!out||!form.elements)return;
 const params=new URLSearchParams({baseline:form.elements.baseline.value,candidate:form.elements.candidate.value,metric:form.elements.metric.value});
 compareState={baseline:params.get('baseline'),candidate:params.get('candidate'),metric:params.get('metric')};
 try{if(params.get('baseline')===params.get('candidate'))throw Error('Choose two different runs.');
  const key=params.toString();compareCache[key]??=await api(`/api/compare?${params}`);
  if(form.isConnected===false)return;out.innerHTML=compareHtml(compareCache[key],currentSection(form)?.visuals!==false);
 }catch(err){out.textContent=err.message;}
}
function autoCompare(){const form=document.getElementById?.('compare-form');if(form&&form.elements&&form.elements.baseline)runCompare(form);}
function pages(){return data.pages??data.layout.pages??data.layout.components.map(id=>({id,title:labels[id],tabs:[{id:'results',title:'Results',sections:[{id,title:labels[id],component:id}]}]}));}
function renderPage(page){
 const tab=page.tabs.find(t=>t.id===activeTabs[page.id])||page.tabs[0];
 activeTabs[page.id]=tab.id;
 const tabs=page.tabs.length<2?'':`<div class="page-tabs" role="tablist" aria-label="${escape(page.title)} tabs">${page.tabs.map(t=>`<button role="tab" aria-selected="${t.id===tab.id}" data-tab="${escape(t.id)}" class="${t.id===tab.id?'active':''}">${escape(t.title)}</button>`).join('')}</div>`;
 return tabs+`<div role="tabpanel" aria-label="${escape(tab.title)}">`+(tab.sections.length?tab.sections.map(section=>{sectionContext=section;try{return `<section class="dashboard-section" data-section-id="${escape(section.id)}">${page.tabs.length<2&&tab.sections.length<2&&section.title.trim().toLowerCase()===String(page.title).trim().toLowerCase()?'':`<h2 class="section-title">${escape(section.title)}</h2>`}${renderers[section.component](traces())}</section>`;}finally{sectionContext=null;}}).join(''):empty('This tab has no sections. Add a section in Customize or through the dashboard skill.'))+'</div>';
}
function currentSection(element){const id=element.closest('.dashboard-section')?.dataset.sectionId;const page=pages().find(p=>p.id===active);const tab=page?.tabs.find(t=>t.id===activeTabs[active]);return tab?.sections.find(s=>s.id===id);}
function renderBrand(){const brand=data.layout.brand||{},logo=brand.logo||{type:'mark',value:'▧'};const mark=logo.type==='image'?`<img class="brand-logo" src="${escape(logo.value)}" alt="${escape(brand.name||data.layout.title)} logo">`:logo.type==='none'?'':`<span class="brand-mark">${escape(logo.value)}</span>`;$('.brand').innerHTML=mark+`<strong>${escape(brand.name??'evalboard')}</strong>${brand.badge=== ''?'':`<span class="brand-badge">${escape(brand.badge??'LAB')}</span>`}`;$('.workspace').innerHTML=`<span class="workspace-initial">${escape((brand.name||data.layout.title).slice(0,1))}</span><span>${escape(data.layout.title)}<small>${escape(brand.tagline??'Local evaluation studio')}</small></span>`;}
// ---- Interpretability (evalboard.interp/v1) -------------------------------------------------
// Every label, token, and title below comes from untrusted artifacts and passes through escape().
// Colors in SVG use presentation attributes and CSS classes only (the CSP forbids inline style).
const interpCache=Object.create(null);
const interpState={run:null,kind:null,pick:Object.create(null),unit:Object.create(null),example:Object.create(null),topn:'50',topk:'3',distNorm:'counts',distY:null};
const interpKinds={layer_series:'Layer metrics',token_layer_map:'Token × layer',logit_lens_table:'Logit lens',head_grid:'Attention heads',distribution:'Distributions',scatter:'Projections',heatmap:'Heatmaps',feature_explorer:'Feature explorer',circuit_graph:'Circuit graph'};
function seeToken(t){return String(t??'').replace(/ /g,'␣').replace(/\n/g,'⏎').replace(/\t/g,'⇥');}
function short(s,n){s=String(s??'');return s.length>n?s.slice(0,n-1)+'…':s;}
function fmtNum(v,d=3){return v===null||v===undefined||!Number.isFinite(v)?'—':(Math.abs(v)>=1000||(v!==0&&Math.abs(v)<.001)?v.toExponential(2):Number(v.toFixed(d)).toString());}
function loadInterp(id){
 if(interpCache[id])return interpCache[id];
 interpCache[id]='loading';
 api('/api/interp/artifact?id='+encodeURIComponent(id)).then(a=>{interpCache[id]=a;render();}).catch(e=>{interpCache[id]={error:e.message};render();});
 return 'loading';
}
function interpViewCurrent(){
 const index=data.interp||[];
 const runs=[...new Set(index.map(a=>a.run_id))];
 if(!runs.includes(interpState.run))interpState.run=runs.at(-1)??null;
 const inRun=index.filter(a=>a.run_id===interpState.run);
 const kinds=Object.keys(interpKinds).filter(k=>inRun.some(a=>a.kind===k));
 if(!kinds.includes(interpState.kind))interpState.kind=kinds[0]??null;
 return {index,runs,inRun,kinds};
}
function selectOptions(items,selected){return items.map(([v,l])=>`<option value="${escape(v)}"${v===selected?' selected':''}>${escape(l)}</option>`).join('');}
function paletteClass(i){return 'ls'+(i%8);}
function lineChart(layers,series,{label='Per-layer values',range=null,axisLabel='Value',refs=[]}={}){
 const W=680,H=320,L=56,R=16,T=16,B=48,xs=layers,xmin=Math.min(...xs),xmax=Math.max(...xs),vals=[...series.flatMap(s=>[...s.values,...(s.lo||[]),...(s.hi||[])]),...refs.map(r=>r.value)].filter(v=>v!==null&&Number.isFinite(v));
 if(!vals.length)return empty('No recorded values in these series.');
 let lo=range?range[0]:Math.min(...vals),hi=range?range[1]:Math.max(...vals);if(lo===hi){lo-=.5;hi+=.5;}
 const X=x=>L+(xmax===xmin?.5:(x-xmin)/(xmax-xmin))*(W-L-R),Y=v=>T+(1-(v-lo)/(hi-lo))*(H-T-B);
 const ticks=[0,.25,.5,.75,1].map(f=>lo+(hi-lo)*f);
 const xt=xs.filter((_,i)=>xs.length<=12||i%Math.ceil(xs.length/12)===0);
 const grid=ticks.map(v=>`<g><line class="grid-line" x1="${L}" x2="${W-R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="chart-value" x="${L-8}" y="${Y(v)+4}" text-anchor="end">${escape(fmtNum(v,2))}</text></g>`).join('')+xt.map(x=>`<text class="chart-value" x="${X(x)}" y="${H-B+18}" text-anchor="middle">${escape(x)}</text>`).join('');
 const refLines=refs.map(r=>`<g><line class="ref-line" x1="${L}" x2="${W-R}" y1="${Y(r.value)}" y2="${Y(r.value)}"/><text class="chart-tick" x="${W-R-4}" y="${Y(r.value)-5}" text-anchor="end">${escape(short(r.label,40))}</text></g>`).join('');
 const lines=series.map((s,si)=>{
  const band=(()=>{if(!s.lo||!s.hi)return '';const runs=[];let cur=[];s.values.forEach((_,i)=>{const lo=s.lo[i],hi=s.hi[i];if(lo===null||hi===null||!Number.isFinite(lo)||!Number.isFinite(hi)){if(cur.length)runs.push(cur);cur=[];}else cur.push(i);});if(cur.length)runs.push(cur);return runs.map(ix=>`<polygon class="ls-band ${paletteClass(si)}" points="${[...ix.map(i=>`${X(xs[i])},${Y(s.hi[i])}`),...[...ix].reverse().map(i=>`${X(xs[i])},${Y(s.lo[i])}`)].join(' ')}"><title>${escape(s.name)} · recorded band</title></polygon>`).join('');})();
  const segs=[];let cur=[];s.values.forEach((v,i)=>{if(v===null||!Number.isFinite(v)){if(cur.length)segs.push(cur);cur=[];}else cur.push([X(xs[i]),Y(v)]);});if(cur.length)segs.push(cur);
  const dots=s.values.map((v,i)=>v===null||!Number.isFinite(v)?'':`<circle class="ls-dot ${paletteClass(si)}" cx="${X(xs[i])}" cy="${Y(v)}" r="3"><title>${escape(s.name)} · layer ${escape(xs[i])}: ${escape(fmtNum(v,4))}${s.n!==undefined?' · n='+escape(s.n):''}</title></circle>`).join('');
  return `<g>${band}${segs.map(p=>`<polyline class="ls-line ${paletteClass(si)}" points="${p.map(q=>q.join(',')).join(' ')}"/>`).join('')}${dots}</g>`;
 }).join('');
 const legend=series.map((s,si)=>`<span class="legend-item"><i class="legend-swatch ${paletteClass(si)}"></i>${escape(s.name)}${s.n!==undefined?` <small class="muted">n=${escape(s.n)}</small>`:''}</span>`).join('');
 return `<figure class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escape(label)}"><title>${escape(label)}</title>${grid}${refLines}<text class="chart-value" x="${(L+W-R)/2}" y="${H-6}" text-anchor="middle">Layer</text><text class="chart-value" transform="rotate(-90 12 ${(T+H-B)/2})" x="12" y="${(T+H-B)/2}" text-anchor="middle">${escape(axisLabel)}</text>${lines}</svg><div class="legend">${legend}</div><figcaption>${escape(label)} · missing values (null) break the line${series.some(s=>s.lo&&s.hi)?' · shaded band = producer-recorded lo/hi':''}</figcaption></figure>`;
}
function layerSeriesView(a){
 const d=a.data;
 const chart=visuals()?lineChart(d.layers,d.series,{label:`${d.metric||a.title}${d.range?` · fixed range ${d.range[0]}–${d.range[1]}`:''}`,range:d.range,axisLabel:d.metric||'Value',refs:d.reference_lines||[]}):'';
 const tbl=table(['Layer',...d.series.map(s=>s.name)],d.layers.map((l,i)=>`<tr><td>${escape(l)}</td>${d.series.map(s=>`<td>${escape(fmtNum(s.values[i],4))}</td>`).join('')}</tr>`));
 return chart+`<details><summary>Exact values</summary>${tbl}</details>`;
}
function heatColor(v,lo,hi,scale){
 if(v===null||v===undefined||!Number.isFinite(v))return null;
 const dark=isDark();
 if(scale==='diverging'){const m=Math.max(Math.abs(lo),Math.abs(hi))||1,f=Math.max(-1,Math.min(1,v/m));
  if(dark)return f>=0?`hsl(8,${35+f*40}%,${20+f*38}%)`:`hsl(215,${30+-f*40}%,${20+-f*38}%)`;
  return f>=0?`hsl(8,70%,${96-f*46}%)`:`hsl(215,65%,${96+f*46}%)`;}
 const f=hi===lo?.5:Math.max(0,Math.min(1,(v-lo)/(hi-lo)));
 return dark?`hsl(${165-f*20},${30+f*30}%,${14+f*50}%)`:`hsl(${165-f*20},${35+f*30}%,${96-f*68}%)`;
}
function heatmapSvg(x,y,grid,lo,hi,scale,label){
 const cols=x.length,rows=y.length,cw=Math.max(8,Math.min(30,560/cols)),ch=Math.max(7,Math.min(20,400/rows)),L=58,T=12,B=cols<=80?72:24,W=L+cols*cw+10,H=T+rows*ch+B;
 const yEvery=Math.ceil(13/ch),xEvery=Math.ceil(15/cw);
 const cells=grid.map((row,r)=>row.map((v,c)=>{const fill=heatColor(v,lo,hi,scale);return `<rect class="${fill?'hm-cell':'hm-cell hm-null'}" x="${L+c*cw}" y="${T+r*ch}" width="${cw}" height="${ch}"${fill?` fill="${fill}"`:''}><title>${escape(y[r])} × ${escape(x[c])}: ${v===null?'not recorded':escape(fmtNum(v,4))}</title></rect>`;}).join('')).join('');
 const yl=y.map((t,r)=>r%yEvery===0?`<text class="chart-tick" x="${L-6}" y="${T+r*ch+ch*.72}" text-anchor="end">${escape(short(t,9))}</text>`:'').join('');
 const xl=cols<=80?x.map((t,c)=>c%xEvery===0?`<text class="chart-tick" transform="rotate(-45 ${L+c*cw+cw/2} ${T+rows*ch+8})" x="${L+c*cw+cw/2}" y="${T+rows*ch+8}" text-anchor="end">${escape(short(seeToken(t),12))}</text>`:'').join(''):'';
 return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escape(label)}"><title>${escape(label)}</title>${cells}${yl}${xl}</svg>`;
}
function heatLegend(lo,hi,scale){const n=9;return `<div class="heat-legend" aria-hidden="true"><span>${escape(fmtNum(lo,3))}</span><svg viewBox="0 0 ${n*18} 10" width="${n*18}" height="10">${Array.from({length:n},(_,i)=>`<rect x="${i*18}" y="0" width="18" height="10" fill="${heatColor(lo+(hi-lo)*i/(n-1),lo,hi,scale)}"/>`).join('')}</svg><span>${escape(fmtNum(hi,3))}</span></div><p class="muted heat-note">Gray dashed cell = not recorded.</p>`;}
function heatmapView(a){
 const d=a.data,flat=d.panels.flatMap(p=>p.values.flat()).filter(v=>v!==null&&Number.isFinite(v));
 if(!flat.length)return empty('Every cell in this heatmap is missing (null).');
 const scale=d.scale==='diverging'?'diverging':'sequential';
 let lo=d.range?d.range[0]:Math.min(...flat),hi=d.range?d.range[1]:Math.max(...flat);
 const panels=d.panels.map(p=>({name:p.name,values:p.values,scale,lo,hi,derived:false}));
 if(d.panels.length===2&&d.show_difference!==false){
  const [p,q]=d.panels,diff=q.values.map((row,r)=>row.map((v,c)=>v===null||p.values[r][c]===null?null:v-p.values[r][c])),m=diff.flat().filter(v=>v!==null);
  if(m.length){const ext=Math.max(...m.map(Math.abs))||1;panels.push({name:`Difference: ${q.name} − ${p.name} (computed here from the two recorded panels)`,values:diff,scale:'diverging',lo:-ext,hi:ext,derived:true});}
 }
 const figs=visuals()?panels.map(p=>`<figure class="chart heatmap"><figcaption class="heat-title">${escape(p.name)}</figcaption>${heatmapSvg(d.x,d.y,p.values,p.lo,p.hi,p.scale,`${a.title} · ${p.name}`)}${heatLegend(p.lo,p.hi,p.scale)}</figure>`).join(''):'';
 const info=`<p class="muted">${escape(d.value_label||'Value')} · rows: ${escape(d.y_label||'y')} · columns: ${escape(d.x_label||'x')}${d.range?` · fixed color range ${escape(d.range[0])}–${escape(d.range[1])}`:' · color range from the recorded data'}</p>`;
 const tables=d.panels.map(p=>`<details><summary>Exact values · ${escape(p.name)}</summary>${table(['',...d.x.map(seeToken)],p.values.map((row,r)=>`<tr><th>${escape(d.y[r])}</th>${row.map(v=>`<td>${escape(fmtNum(v,4))}</td>`).join('')}</tr>`))}</details>`).join('');
 return info+`<div class="heat-grid">${figs}</div>`+tables;
}
function tokenClass(v,max){if(v===null||!Number.isFinite(v)||!max)return 'tok';const f=Math.min(1,Math.abs(v)/max),b=Math.round(f*8);return b?`tok ${v>0?'p':'n'}${b}`:'tok';}
function tokenRow(ex,max){
 const traceBtn=ex.trace_id&&data.traces.some(t=>t.id===ex.trace_id)?` <button class="nowrap" data-trace="${escape(ex.trace_id)}">Open trace ↗</button>`:'';
 const peak=Math.max(...ex.activations.filter(v=>v!==null&&Number.isFinite(v)).map(Math.abs),0);
 return `<div class="tok-example"><div class="tok-meta"><div class="tok-meta-text"><strong>${escape(ex.case_id||'example')}</strong><small class="muted">peak |act| ${escape(fmtNum(peak))}${ex.note?' · '+escape(ex.note):''}</small></div>${traceBtn}</div><div class="tok-line" role="text">${ex.tokens.map((t,i)=>`<span class="${tokenClass(ex.activations[i],max)}" title="${escape(seeToken(t))}: ${ex.activations[i]===null?'not recorded':escape(fmtNum(ex.activations[i],4))}">${escape(t)}</span>`).join('')}</div></div>`;
}
function histogramSvg(h,ymax,label){
 const W=360,H=130,L=8,B=22,T=6,n=h.counts.length,bw=(W-L-8)/n,lo=h.edges[0],hi=h.edges[n];
 const bars=h.counts.map((c,i)=>{const bh=ymax?c/ymax*(H-B-T):0;return `<rect class="chart-bar" x="${L+i*bw+.5}" y="${H-B-bh}" width="${Math.max(1,bw-1)}" height="${bh}"><title>[${escape(fmtNum(h.edges[i]))}, ${escape(fmtNum(h.edges[i+1]))}): ${escape(c)}</title></rect>`;}).join('');
 return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escape(label)}"><title>${escape(label)}</title>${bars}<line class="grid-line" x1="${L}" x2="${W-8}" y1="${H-B}" y2="${H-B}"/><text class="chart-value" x="${L}" y="${H-6}">${escape(fmtNum(lo))}</text><text class="chart-value" x="${W-8}" y="${H-6}" text-anchor="end">${escape(fmtNum(hi))}</text></svg>`;
}
function logitEffects(e){
 if(!e)return '<p class="muted">Logit effects not recorded for this condition.</p>';
 const side=(items,name,tone)=>items&&items.length&&!visuals()?table(['Token',name+' effect'],items.slice(0,10).map(i=>`<tr><td>${escape(seeToken(i.token))}</td><td>${escape(fmtNum(i.value,4))}</td></tr>`)):items&&items.length?barChart(items.slice(0,10).map(i=>({label:seeToken(i.token),value:Math.abs(i.value),tone,note:name+' · '+fmtNum(i.value,4)})),{label:`${name} tokens${e.method?' · '+e.method:''}`,format:v=>fmtNum(v,3)}):`<p class="muted">No ${escape(name)} tokens recorded.</p>`;
 return `<div class="logit-effects"><h5>Promoted</h5>${side(e.promoted,'Promoted','')}<h5>Suppressed</h5>${side(e.suppressed,'Suppressed','negative')}</div>`;
}
function featureExplorerView(a){
 const d=a.data;if(!d.units.length)return empty('This artifact lists no units.');
 const key=a.id,unitIds=d.units.map(u=>u.id);
 if(!unitIds.includes(interpState.unit[key]))interpState.unit[key]=unitIds[0];
 const u=d.units.find(x=>x.id===interpState.unit[key]);
 const opts=selectOptions(d.units.map(x=>[x.id,`L${x.layer} · ${x.kind.replace('_',' ')}${x.index!==undefined?' #'+x.index:''}${x.label?' — '+short(x.label,48):''}`]),u.id);
 const peaks=u.conditions.flatMap(c=>(c.examples||[]).flatMap(e=>e.activations)).filter(v=>v!==null&&Number.isFinite(v)).map(Math.abs);
 const max=Math.max(...peaks,0),ymax=Math.max(...u.conditions.map(c=>c.histogram?Math.max(...c.histogram.counts):0),0);
 const cols=u.conditions.map(c=>{
  const s=c.stats||{};
  const stats=Object.keys(s).length?`<p class="muted">${Object.entries(s).map(([k,v])=>`${escape(k)} ${escape(fmtNum(v))}`).join(' · ')}</p>`:'';
  const hist=c.histogram&&visuals()?`<h4>Activation histogram</h4>${histogramSvg(c.histogram,ymax,`${u.id} · ${c.name} activation histogram`)}`:(c.histogram?'':'<p class="muted">Histogram not recorded.</p>');
  const ex=(c.examples||[]).length?(c.examples||[]).map(e=>tokenRow(e,max)).join(''):'<p class="muted">No top-activating examples recorded.</p>';
  return `<div class="cond"><h3>${escape(c.name)}</h3>${stats}${hist}<h4>Top-activating examples</h4>${ex}<h4>Direct logit effects</h4>${logitEffects(c.logit_effects)}</div>`;
 }).join('');
 return `<div class="interp-controls"><label>Unit <select data-interp-unit="${escape(key)}">${opts}</select></label><span class="muted">${escape(d.activation_label||'Activation')} · token shading is shared across conditions (max |act| ${escape(fmtNum(max))}); ${'<span class="tok p6">positive</span> <span class="tok n6">negative</span>'}</span></div><div class="cond-grid cols-${Math.min(3,u.conditions.length||1)}">${cols}</div>`;
}
function circuitGraphView(a){
 const d=a.data,edgeKind=d.edge_kind;
 const notes={measured:'Edges are measured attributions recorded by the producer.',layer_adjacent:'Edges are NOT measured connections. They link in-circuit nodes in adjacent layers and are weighted by the product of the two node scores (CircuitKIT’s visualisation). Read the node scores, not the edges.',derived:'Edges were derived by the producer, not measured; see the artifact notes.'};
 const topn=interpState.topn,limit=topn==='all'?Infinity:Number(topn);
 const ranked=[...d.nodes].sort((p,q)=>Math.abs(q.score)-Math.abs(p.score)),kept=ranked.slice(0,limit),keptIds=new Set(kept.map(n=>n.id));
 const edges=(d.edges||[]).filter(e=>keptIds.has(e.source)&&keptIds.has(e.target)).sort((p,q)=>Math.abs(q.weight)-Math.abs(p.weight)).slice(0,400);
 const callout=`<div class="callout${edgeKind==='measured'?'':' warn'}"><strong>Edges: ${escape(edgeKind)}</strong><br>${escape(notes[edgeKind])}${d.algorithm?`<br><small>${escape(d.algorithm)}${d.task?' · task '+escape(d.task):''}</small>`:''}</div>`;
 const controls=`<div class="interp-controls"><label>Show top nodes by |score| <select data-interp-topn>${selectOptions([['25','25'],['50','50'],['100','100'],['250','250'],['all','all ('+d.nodes.length+')']],topn)}</select></label><span class="muted">${kept.length} of ${d.nodes.length} nodes · ${edges.length} edges drawn${(d.edges||[]).length>edges.length?' (filtered)':''} · ${escape(d.score_label||'score')}</span></div>`;
 let chart='';
 if(visuals()&&kept.length){
  const layers=kept.map(n=>n.layer),lmin=Math.min(...layers),lmax=Math.max(...layers),rowOf=n=>n.head===undefined||n.head===null?0:n.head+1,rmax=Math.max(...kept.map(rowOf),1),W=900,H=Math.max(300,Math.min(620,(rmax+1)*34+70)),L=48,R=36,T=28,B=36;
  const X=l=>L+(lmax===lmin?.5:(l-lmin)/(lmax-lmin))*(W-L-R),Y=r=>T+r/(rmax||1)*(H-T-B),smax=Math.max(...kept.map(n=>Math.abs(n.score)),1e-12),wmax=Math.max(...edges.map(e=>Math.abs(e.weight)),1e-12),byId=Object.fromEntries(kept.map(n=>[n.id,n]));
  const ls=edges.map(e=>{const s=byId[e.source],t=byId[e.target];return `<line class="cg-edge ${edgeKind==='measured'?'':'cg-derived'}" x1="${X(s.layer)}" y1="${Y(rowOf(s))}" x2="${X(t.layer)}" y2="${Y(rowOf(t))}" stroke-width="${.5+2.5*Math.abs(e.weight)/wmax}" opacity="${.15+.6*Math.abs(e.weight)/wmax}"><title>${escape(e.source)} → ${escape(e.target)}: ${escape(fmtNum(e.weight,4))} (${escape(edgeKind)})</title></line>`;}).join('');
  const ns=kept.map(n=>`<circle class="cg-node cg-${escape(n.type==='mlp'||/^(m\d|MLP)/.test(n.id)?'mlp':(n.type==='attn'||n.head!==undefined&&n.head!==null?'attn':'other'))}${n.in_circuit===false?' cg-out':''}" cx="${X(n.layer)}" cy="${Y(rowOf(n))}" r="${3+9*Math.abs(n.score)/smax}"><title>${escape(n.label||n.id)} · layer ${escape(n.layer)}${n.head!==undefined&&n.head!==null?' · head '+escape(n.head):''} · score ${escape(fmtNum(n.score,5))}${n.in_circuit===false?' · pruned / outside circuit':''}</title></circle>`).join('');
  const lset=[...new Set(layers)].sort((p,q)=>p-q),lstep=Math.ceil(lset.length/14),ax=lset.filter((_,i)=>i%lstep===0).map(l=>`<text class="chart-tick" x="${X(l)}" y="${H-12}" text-anchor="middle">L${escape(l)}</text>`).join('');
  const top=kept.slice(0,6).map(n=>`<text class="cg-label" x="${X(n.layer)}" y="${Y(rowOf(n))-(3+9*Math.abs(n.score)/smax)-4}" text-anchor="middle">${escape(short(n.label||n.id,12))}</text>`).join('');
  chart=`<figure class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escape(a.title)} circuit graph"><title>${escape(a.title)} · x: layer, y: MLP row then attention head index, size: |score|</title>${ls}${ns}${top}${ax}</svg><figcaption>x = layer · top row = MLP · lower rows = attention head index · circle area ∝ |score| · dashed outline = outside circuit</figcaption></figure>`;
 }
 const nodeRows=kept.slice(0,50).map((n,i)=>`<tr><td>${i+1}</td><td>${escape(n.label||n.id)}</td><td>${escape(n.layer)}</td><td>${n.head===undefined||n.head===null?'—':escape(n.head)}</td><td>${escape(fmtNum(n.score,5))}</td><td>${n.in_circuit===false?'no':'yes'}</td></tr>`),nodeHead=['Rank','Node','Layer','Head','Score','In circuit'];
 return callout+controls+chart+`<details open><summary>Top ${Math.min(12,nodeRows.length)} nodes by |score|</summary>${table(nodeHead,nodeRows.slice(0,12))}</details>${nodeRows.length>12?`<details><summary>Nodes 13–${nodeRows.length} (up to 50 listed)</summary>${table(nodeHead,nodeRows.slice(12))}</details>`:''}`;
}
function linkedTraces(a){
 const ids=(a.trace_ids||[]).filter(id=>data.traces.some(t=>t.id===id)).slice(0,12);
 const same=new Map();ids.forEach(id=>{const c=data.traces.find(t=>t.id===id).case_id;same.set(c,(same.get(c)||0)+1);});
 return ids.length?`<div class="interp-links"><strong>Linked traces</strong> ${ids.map(id=>{const t=data.traces.find(x=>x.id===id),dup=same.get(t.case_id)>1;return `<button class="nowrap" data-trace="${escape(id)}" title="${escape(id)}">${escape(short(t.case_id,40))}${dup?' · '+escape(midEllipsis(runLabel(t.run_id).split(':').pop().replace(/^eval-/,''),22)):''} ↗</button>`;}).join(' ')}</div>`:'';
}
// ---- Extended mechanistic-interpretability views ----------------------------------------------
// token_layer_map, logit_lens_table, head_grid, distribution, scatter. Anything computed in the browser from recorded
// values (layer means, differences, densities, centroids, top-k lists) is labelled "computed here".
function fmtTick(v){return v===null||!Number.isFinite(v)?'—':Math.abs(v)>=1000?Math.round(v).toLocaleString('en-US'):Math.abs(v)>=10?String(Math.round(v)):fmtNum(v,2);}
function computedNote(text){return `<p class="muted computed-note"><span class="badge">computed here</span> ${escape(text)}</p>`;}
function layerLabels(layers){return layers.map(l=>'L'+l);}
function finiteVals(a){return a.filter(v=>v!==null&&v!==undefined&&Number.isFinite(v));}
function meanOf(a){const v=finiteVals(a);return v.length?v.reduce((p,q)=>p+q,0)/v.length:null;}
function tokenStrip(tokens,values,max,title){
 return `<div class="tok-example"><div class="tok-meta"><div class="tok-meta-text"><strong>${escape(title)}</strong><small class="muted">max |value| ${escape(fmtNum(max))}</small></div></div><div class="tok-line" role="text">${tokens.map((t,i)=>`<span class="${tokenClass(values[i],max)}" title="${escape(seeToken(t))}: ${values[i]===null?'not recorded':escape(fmtNum(values[i],4))}">${escape(t)}</span>`).join('')}</div></div>`;
}
function tokenLayerMapView(a){
 const d=a.data,ex0=d.examples,key=a.id;
 if(!ex0.some(e=>e.id===interpState.example[key]))interpState.example[key]=ex0[0].id;
 const ex=ex0.find(e=>e.id===interpState.example[key]);
 const picker=ex0.length>1?`<div class="interp-controls"><label>Prompt <select data-interp-example="${escape(key)}">${selectOptions(ex0.map(e=>[e.id,e.label||e.case_id||e.id]),ex.id)}</select></label></div>`:'';
 const heat=heatmapView({title:a.title,data:{x:ex.tokens,y:layerLabels(d.layers),panels:ex.panels,range:d.range,scale:d.scale,value_label:d.value_label||'Activation',x_label:'prompt token',y_label:'layer'}});
 // Layer-mean per token: shading strip so a reader can scan tokens before reading the matrix.
 const means=ex.panels.map(p=>({name:p.name,vals:ex.tokens.map((_,c)=>meanOf(p.values.map(row=>row[c])))}));
 const max=Math.max(...means.flatMap(m=>finiteVals(m.vals).map(Math.abs)),1e-12);
 const strips=visuals()?`<h4>Per-token mean over layers</h4>${means.map(m=>tokenStrip(ex.tokens,m.vals,max,m.name)).join('')}${computedNote('Mean of the recorded layer cells for each token (nulls skipped); shading is shared across panels.')}`:'';
 const link=ex.trace_id&&data.traces.some(t=>t.id===ex.trace_id)?`<p><button class="nowrap" data-trace="${escape(ex.trace_id)}">Open trace ↗</button></p>`:'';
 return picker+link+strips+`<h4>Token × layer matrix</h4>`+heat;
}
function lensCell(row,k,target){
 if(!row)return '<td class="lens-empty">—</td>';
 const top=row.top.slice(0,k),hit=target&&row.top[0]&&row.top[0].token===target;
 return `<td class="lens-cell${hit?' lens-hit':''}" title="${escape(row.top.map((t,i)=>`${i+1}. ${seeToken(t.token)} ${fmtNum(t.value,4)}`).join('\n'))}">${top.map((t,i)=>`<span class="lens-tok${i===0?' lens-top':''}">${escape(seeToken(t.token))} <small>${escape(fmtNum(t.value,3))}</small></span>`).join('')}</td>`;
}
function logitLensTableView(a){
 const d=a.data,target=d.target?.token??null,k=Math.min(Number(interpState.topk)||1,5),panels=d.panels;
 const layers=[...new Set(panels.flatMap(p=>p.layers.map(r=>r.layer)))].sort((p,q)=>p-q);
 const byLayer=panels.map(p=>new Map(p.layers.map(r=>[r.layer,r])));
 const controls=`<div class="interp-controls"><label>Tokens per cell <select data-interp-topk>${selectOptions([['1','top-1'],['3','top-3'],['5','top-5']],String(k))}</select></label><span class="muted">${escape(d.value_label||'value')} · position: ${escape(d.position_label||'final prompt position')}${target!==null?` · target token <code>${escape(seeToken(target))}</code> (green cells = top-1 is the target)`:''}</span></div>`;
 const head=`<tr><th rowspan="2">Layer</th>${panels.map(p=>`<th colspan="${target!==null?3:1}" class="lens-group">${escape(p.name)}</th>`).join('')}</tr>${target!==null?`<tr>${panels.map(()=>'<th>Top tokens</th><th>Target rank</th><th>Target value</th>').join('')}</tr>`:''}`;
 const body=layers.map(l=>`<tr><th>L${escape(l)}</th>${panels.map((p,i)=>{const row=byLayer[i].get(l);return lensCell(row,k,target)+(target!==null?`<td>${row&&row.target_rank!==undefined&&row.target_rank!==null?escape(row.target_rank):'—'}</td><td>${row&&row.target_value!==undefined&&row.target_value!==null?escape(fmtNum(row.target_value,4)):'—'}</td>`:'');}).join('')}</tr>`).join('');
 const hasVal=target!==null&&panels.some(p=>p.layers.some(r=>r.target_value!==undefined&&r.target_value!==null));
 const chart=visuals()&&hasVal?lineChart(layers,panels.map((p,i)=>({name:p.name,values:layers.map(l=>{const r=byLayer[i].get(l);return r&&r.target_value!==undefined?r.target_value:null;})})),{label:`Target token ${seeToken(target)} · ${d.value_label||'value'} by layer`,axisLabel:d.value_label||'value'}):'';
 const firsts=target!==null?panels.map(p=>{const r=[...p.layers].sort((x,y)=>x.layer-y.layer).find(r=>r.top[0].token===target);return `${escape(p.name)}: ${r?'layer '+escape(r.layer):'never top-1'}`;}).join(' · '):'';
 return controls+chart+(firsts?computedNote(`First layer where the target is the top-1 token — ${firsts}`):'')+`<div class="table-wrap"><table class="lens-table"><thead>${head}</thead><tbody>${body}</tbody></table></div>`;
}
function topCells(layers,values,n){
 const cells=[];values.forEach((row,r)=>row.forEach((v,c)=>{if(v!==null&&Number.isFinite(v))cells.push({r,c,v});}));
 return cells.sort((p,q)=>Math.abs(q.v)-Math.abs(p.v)).slice(0,n);
}
function headGridView(a){
 const d=a.data,heads=Array.from({length:d.n_heads},(_,i)=>'H'+i);
 const heat=heatmapView({title:a.title,data:{x:heads,y:layerLabels(d.layers),panels:d.panels,range:d.range,scale:d.scale,value_label:d.value_label||'Importance',x_label:'attention head',y_label:'layer',show_difference:d.show_difference}});
 const tops=d.panels.map(p=>`<div><h4>Top 10 heads · ${escape(p.name)}</h4>${table(['Rank','Head',d.value_label||'Value'],topCells(d.layers,p.values,10).map((c,i)=>`<tr><td>${i+1}</td><td>L${escape(d.layers[c.r])}.H${c.c}</td><td>${escape(fmtNum(c.v,4))}</td></tr>`))}</div>`).join('');
 let diff='';
 if(d.panels.length===2&&d.show_difference!==false){
  const [p,q]=d.panels,dv=q.values.map((row,r)=>row.map((v,c)=>v===null||p.values[r][c]===null?null:v-p.values[r][c]));
  diff=`<div><h4>Largest changes · ${escape(q.name)} − ${escape(p.name)}</h4>${table(['Rank','Head','Change'],topCells(d.layers,dv,10).map((c,i)=>`<tr><td>${i+1}</td><td>L${escape(d.layers[c.r])}.H${c.c}</td><td>${escape(fmtNum(c.v,4))}</td></tr>`))}${computedNote('Difference of the two recorded panels, per head.')}</div>`;
 }
 return heat+`<div class="head-tops">${tops}${diff}</div>`;
}
function distributionView(a){
 const d=a.data,norm=interpState.distNorm==='density'?'density':'counts',yscale=interpState.distY||d.y_scale||'linear',n=d.edges.length-1;
 const widths=d.edges.slice(1).map((e,i)=>e-d.edges[i]);
 const series=d.series.map(s=>{const total=s.counts.reduce((p,q)=>p+q,0)||1;return {...s,vals:s.counts.map((c,i)=>norm==='density'?c/total/widths[i]:c)};});
 const controls=`<div class="interp-controls"><label>Y values <select data-interp-dist-norm>${selectOptions([['counts','counts (recorded)'],['density','density (computed here)']],norm)}</select></label><label>Y axis <select data-interp-dist-y>${selectOptions([['linear','linear'],['log','log (log1p)']],yscale)}</select></label><span class="muted">${escape(d.value_label||'value')} · ${n} shared bins</span></div>`;
 let chart='';
 if(visuals()){
  const W=720,H=330,L=60,R=18,T=18,B=54,xmin=d.edges[0],xmax=d.edges[n],X=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R),f=v=>yscale==='log'?Math.log1p(v):v,ymax=Math.max(...series.flatMap(s=>s.vals.map(f)),1e-12),Y=v=>T+(1-f(v)/ymax)*(H-T-B);
  const yt=[0,.25,.5,.75,1].map(q=>{const fv=ymax*q,raw=yscale==='log'?Math.expm1(fv):fv;return `<g><line class="grid-line" x1="${L}" x2="${W-R}" y1="${T+(1-q)*(H-T-B)}" y2="${T+(1-q)*(H-T-B)}"/><text class="chart-tick" x="${L-8}" y="${T+(1-q)*(H-T-B)+4}" text-anchor="end">${escape(fmtTick(raw))}</text></g>`;}).join('');
  const xt=Array.from({length:6},(_,i)=>xmin+(xmax-xmin)*i/5).map(v=>`<text class="chart-tick" x="${X(v)}" y="${H-B+18}" text-anchor="middle">${escape(fmtNum(v,3))}</text>`).join('');
  const bars=series.map((s,si)=>s.vals.map((v,i)=>v>0?`<rect class="dist-bar ${paletteClass(si)}" x="${X(d.edges[i])}" y="${Y(v)}" width="${Math.max(1,X(d.edges[i+1])-X(d.edges[i])-1)}" height="${H-B-Y(v)}"><title>${escape(s.name)} · [${escape(fmtNum(d.edges[i]))}, ${escape(fmtNum(d.edges[i+1]))}): ${escape(fmtNum(v,4))}</title></rect>`:'').join('')).join('');
  const marks=(d.markers||[]).filter(m=>m.value>=xmin&&m.value<=xmax).map(m=>`<g><line class="dist-marker" x1="${X(m.value)}" x2="${X(m.value)}" y1="${T}" y2="${H-B}"/><text class="chart-tick halo" x="${X(m.value)+5}" y="${T+11}">${escape(short(m.label,24))}</text></g>`).join('');
  const legend=series.map((s,si)=>`<span class="legend-item"><i class="legend-swatch ${paletteClass(si)}"></i>${escape(s.name)}${s.n!==undefined?` <small class="muted">n=${escape(s.n)}</small>`:''}</span>`).join('');
  chart=`<figure class="chart"><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escape(a.title)}"><title>${escape(a.title)}</title>${yt}${bars}${marks}${xt}<text class="chart-tick" x="${(L+W-R)/2}" y="${H-8}" text-anchor="middle">${escape(d.x_label||d.value_label||'value')}</text></svg><div class="legend">${legend}</div><figcaption>Overlaid histograms on shared bin edges · ${norm==='density'?'density = count / total / bin width':'recorded counts'} · ${yscale==='log'?'log1p y-axis':'linear y-axis'}</figcaption></figure>`;
 }
 const rows=series.map(s=>{const stats=Object.entries(s.stats||{}).map(([k,v])=>`${escape(k)} ${escape(fmtNum(v))}`).join(' · ');return `<tr><td>${escape(s.name)}</td><td>${s.n!==undefined?escape(s.n):'—'}</td><td>${stats||'—'}</td></tr>`;});
 return controls+chart+(norm==='density'?computedNote('Densities are normalised from the recorded counts in the browser.'):'')+`<details open><summary>Series</summary>${table(['Series','n','Recorded stats'],rows)}</details><details><summary>Exact bin counts</summary>${table(['Bin',...d.series.map(s=>s.name)],d.edges.slice(0,-1).map((e,i)=>`<tr><td>[${escape(fmtNum(e))}, ${escape(fmtNum(d.edges[i+1]))})</td>${d.series.map(s=>`<td>${escape(s.counts[i])}</td>`).join('')}</tr>`))}</details>`;
}
function scatterView(a){
 const d=a.data,LIMIT=4000,ev=d.explained_variance;
 const pts=d.panels.flatMap(p=>p.groups.flatMap(g=>g.points));
 if(!pts.length)return empty('This artifact has no points.');
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),px=(Math.max(...xs)-Math.min(...xs))*.05||1,py=(Math.max(...ys)-Math.min(...ys))*.05||1;
 const x0=Math.min(...xs)-px,x1=Math.max(...xs)+px,y0=Math.min(...ys)-py,y1=Math.max(...ys)+py;
 const groupNames=[...new Set(d.panels.flatMap(p=>p.groups.map(g=>g.name)))];
 let subsampled=false;
 const figs=d.panels.map(p=>{
  const W=460,H=420,L=48,R=14,T=14,B=44,X=v=>L+(v-x0)/(x1-x0)*(W-L-R),Y=v=>T+(1-(v-y0)/(y1-y0))*(H-T-B);
  const grid=[0,.5,1].map(q=>`<g><line class="grid-line" x1="${L}" x2="${W-R}" y1="${T+q*(H-T-B)}" y2="${T+q*(H-T-B)}"/><text class="chart-tick" x="${L-6}" y="${T+q*(H-T-B)+4}" text-anchor="end">${escape(fmtNum(y1-(y1-y0)*q,2))}</text><line class="grid-line" x1="${L+q*(W-L-R)}" x2="${L+q*(W-L-R)}" y1="${T}" y2="${H-B}"/><text class="chart-tick" x="${L+q*(W-L-R)}" y="${H-B+16}" text-anchor="middle">${escape(fmtNum(x0+(x1-x0)*q,2))}</text></g>`).join('');
  const dots=p.groups.map(g=>{const gi=groupNames.indexOf(g.name),step=Math.max(1,Math.ceil(g.points.length/LIMIT));if(step>1)subsampled=true;return g.points.map((pt,i)=>i%step===0?`<circle class="sc-dot ${paletteClass(gi)}" cx="${X(pt[0])}" cy="${Y(pt[1])}" r="3"><title>${escape(g.name)}${g.labels?' · '+escape(g.labels[i]):''} (${escape(fmtNum(pt[0],3))}, ${escape(fmtNum(pt[1],3))})</title></circle>`:'').join('');}).join('');
  const cents=p.groups.filter(g=>g.points.length).map(g=>{const gi=groupNames.indexOf(g.name),cx=meanOf(g.points.map(q=>q[0])),cy=meanOf(g.points.map(q=>q[1]));return `<circle class="sc-centroid ${paletteClass(gi)}" cx="${X(cx)}" cy="${Y(cy)}" r="7"><title>${escape(g.name)} centroid (computed here): (${escape(fmtNum(cx,3))}, ${escape(fmtNum(cy,3))})</title></circle>`;}).join('');
  const xl=(d.x_label||'dim 1')+(ev&&ev[0]!==null?` (${(ev[0]*100).toFixed(1)}% var)`:''),yl=(d.y_label||'dim 2')+(ev&&ev[1]!==null?` (${(ev[1]*100).toFixed(1)}% var)`:'');
  return `<figure class="chart scatter"><figcaption class="heat-title">${escape(p.name)}</figcaption><svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${escape(a.title)} · ${escape(p.name)}"><title>${escape(a.title)} · ${escape(p.name)}</title>${grid}${dots}${cents}<text class="chart-tick" x="${(L+W-R)/2}" y="${H-6}" text-anchor="middle">${escape(xl)}</text><text class="chart-tick" transform="rotate(-90 12 ${(T+H-B)/2})" x="12" y="${(T+H-B)/2}" text-anchor="middle">${escape(yl)}</text></svg></figure>`;
 }).join('');
 const legend=groupNames.map((n,i)=>`<span class="legend-item"><i class="legend-swatch ${paletteClass(i)}"></i>${escape(n)}</span>`).join('');
 const sep=d.panels.filter(p=>p.groups.length>=2&&p.groups[0].points.length&&p.groups[1].points.length).map(p=>{const [g,h]=p.groups,dx=meanOf(g.points.map(q=>q[0]))-meanOf(h.points.map(q=>q[0])),dy=meanOf(g.points.map(q=>q[1]))-meanOf(h.points.map(q=>q[1]));return `<tr><td>${escape(p.name)}</td><td>${escape(g.name)} vs ${escape(h.name)}</td><td>${escape(fmtNum(Math.hypot(dx,dy),4))}</td></tr>`;});
 return `<p class="muted"><strong>Method:</strong> ${escape(d.method)}</p>`+(visuals()?`<div class="heat-grid scatter-grid">${figs}</div><div class="legend">${legend}<span class="legend-item"><i class="sc-ring"></i>ring = group centroid (computed here)</span></div>${subsampled?computedNote(`Panels with more than ${LIMIT} points per group are drawn with an even stride; the stored artifact is unchanged.`):''}`:'')+(sep.length?`<details open><summary>Centroid separation</summary>${table(['Panel','Groups','Distance between centroids'],sep)}${computedNote('Euclidean distance between the first two groups’ mean positions in this 2-D projection. It is a property of the projection, not of the model’s full activation space.')}</details>`:'')+`<details><summary>Point counts</summary>${table(['Panel','Group','Points'],d.panels.flatMap(p=>p.groups.map(g=>`<tr><td>${escape(p.name)}</td><td>${escape(g.name)}</td><td>${escape(g.points.length)}</td></tr>`)))}</details>`;
}
const interpRenderers={layer_series:layerSeriesView,token_layer_map:tokenLayerMapView,logit_lens_table:logitLensTableView,head_grid:headGridView,distribution:distributionView,scatter:scatterView,heatmap:heatmapView,feature_explorer:featureExplorerView,circuit_graph:circuitGraphView};
function provenanceBadge(a){
 const labels={measured:'Measured by the producer from model activations',derived:'Derived (post-processed, projected, or estimated) — not a raw measurement',mixed:'Mixed: some values measured, some derived'};
 return a.provenance?`<p class="muted"><span class="badge${a.provenance==='measured'?'':' error'}">${escape(a.provenance)}</span> ${escape(labels[a.provenance])}${a.provenance_note?' · '+escape(a.provenance_note):''}</p>`:'';
}
function interpView(){
 const {index,runs,inRun,kinds}=interpViewCurrent();
 if(!index.length)return panel('Mechanistic interpretability','Probes, logit lens, activations, attention heads, distributions, projections, feature explorer, and circuit graphs',empty('No interpretability artifacts yet. Ingest evalboard.interp/v1 artifacts (POST /api/interp, or the ingest_interp_artifacts MCP tool); see the interpretability reference in the skill.'));
 const synthetic=index.some(a=>a.synthetic)?'<span class="badge error">SYNTHETIC FIXTURE — not model output</span> ':'';
 const inKind=inRun.filter(a=>a.kind===interpState.kind);
 if(!inKind.some(a=>a.id===interpState.pick[interpState.kind]))interpState.pick[interpState.kind]=inKind[0]?.id;
 const sel=inKind.find(a=>a.id===interpState.pick[interpState.kind]);
 const head=`<div class="interp-controls">${synthetic}<label>Interp run <select data-interp-run>${selectOptions(runs.map(r=>[r,interpRunLabel(r)]),interpState.run)}</select></label></div><div class="page-tabs" role="tablist" aria-label="Interpretability views">${kinds.map(k=>`<button role="tab" aria-selected="${k===interpState.kind}" data-interp-kind="${escape(k)}" class="${k===interpState.kind?'active':''}">${escape(interpKinds[k])} <small>${inRun.filter(a=>a.kind===k).length}</small></button>`).join('')}</div>`;
 const picker=inKind.length>1?`<div class="interp-controls"><label>Artifact <select data-interp-artifact>${selectOptions(inKind.map(a=>[a.id,a.title+(a.phase?' · '+a.phase:'')]),sel.id)}</select></label></div>`:'';
 const art=sel?loadInterp(sel.id):null;
 let body;
 if(!sel)body=empty('No artifacts of this kind in the selected run.');
 else if(art==='loading')body=empty('Loading artifact…');
 else if(art.error)body=empty('Could not load artifact: '+art.error);
 else body=`<h3 class="interp-title">${escape(art.title)}</h3><p class="muted">${escape(art.model||'')}${art.phase?' · '+escape(art.phase):''}${art.notes?' · '+escape(art.notes):''}</p>${provenanceBadge(art)}${interpRenderers[art.kind](art)}${linkedTraces(art)}`;
 return `<article class="panel interp-panel"><div class="panel-head"><div><h2>Mechanistic interpretability</h2><p class="muted">Recorded artifacts only; missing values stay missing. ${inRun.length} artifacts in this run.</p></div></div>${head}${picker}<div class="panel-body">${body}</div></article>`;
}
function interpRefs(t){
 const ids=[...(t.metadata?.interp||[]).map(r=>({id:r.artifact_id,label:r.label}))];
 (data.interp||[]).filter(a=>(a.trace_ids||[]).includes(t.id)).forEach(a=>{if(!ids.some(r=>r.id===a.id))ids.push({id:a.id});});
 return ids;
}
function interpTraceSection(t){
 const refs=interpRefs(t);if(!refs.length)return '';
 return `<h3>Interpretability views</h3>${refs.map(r=>{const a=(data.interp||[]).find(x=>x.id===r.id);return a?`<div class="callout"><strong>${escape(r.label||a.title)}</strong> <span class="badge">${escape(interpKinds[a.kind]||a.kind)}</span> <button data-interp-open="${escape(a.id)}">Open in Interpretability ↗</button></div>`:`<div class="callout"><strong>${escape(r.label||r.id)}</strong> <small class="muted">referenced artifact not ingested</small></div>`;}).join('')}`;
}
function openInterp(id){
 const a=(data.interp||[]).find(x=>x.id===id);if(!a)return;
 const page=pages().find(p=>p.tabs.some(tb=>tb.sections.some(s=>s.component==='interp')));if(!page){notice('No page has an interpretability section. Enable it in Customize.',true);return;}
 const tab=page.tabs.find(tb=>tb.sections.some(s=>s.component==='interp'));
 interpState.run=a.run_id;interpState.kind=a.kind;interpState.pick[a.kind]=a.id;active=page.id;activeTabs[page.id]=tab.id;
 if($('#trace-dialog').open)$('#trace-dialog').close();
 render();
}
document.addEventListener('change',e=>{const el=e.target;
 if(el.dataset?.interpRun!==undefined){interpState.run=el.value;render();}
 else if(el.dataset?.interpArtifact!==undefined){interpState.pick[interpState.kind]=el.value;render();}
 else if(el.dataset?.interpUnit!==undefined){interpState.unit[el.dataset.interpUnit]=el.value;render();}
 else if(el.dataset?.interpTopn!==undefined){interpState.topn=el.value;render();}
 else if(el.dataset?.interpExample!==undefined){interpState.example[el.dataset.interpExample]=el.value;render();}
 else if(el.dataset?.interpTopk!==undefined){interpState.topk=el.value;render();}
 else if(el.dataset?.interpDistNorm!==undefined){interpState.distNorm=el.value;render();}
 else if(el.dataset?.interpDistY!==undefined){interpState.distY=el.value;render();}
});
document.addEventListener('click',e=>{const k=e.target.closest?.('[data-interp-kind]'),o=e.target.closest?.('[data-interp-open]');
 if(k){interpState.kind=k.dataset.interpKind;render();}
 if(o)openInterp(o.dataset.interpOpen);
});
const renderers = {
 interp:interpView,
 metrics:metricView,
 overview(ts){if(!metrics().length)return `<div class="callout">Built from trace evidence. Execution metrics are available below; quality scores require attributed evaluator results.</div>${metricView(ts)}${panel('Recent traces','Inspect recorded execution evidence',traceTable(ts.slice(-5).reverse()))}`;const metric=selectedMetric(), scored=ts.filter(t=>latest(t,metric)),allRuns=runScores(ts,metric),runs=allRuns.filter(r=>r.value!==null),unscoredRuns=allRuns.length-runs.length;const cards=[['Traces',ts.length,'Recorded evaluation cases'],['Runs',new Set(ts.map(t=>t.run_id)).size,'Distinct experiment runs'],[metricLabel(metric)||'Score',pct(average(scored.map(t=>latest(t,metric).value))),`${scored.length}/${ts.length} cases scored · latest score`],['Reported cost',ts.some(t=>traceMetricValue(t,'cost_usd')!==null)?`$${ts.reduce((a,t)=>a+(traceMetricValue(t,'cost_usd')??0),0).toFixed(3)}`:'Not recorded',ts.some(t=>traceMetricValue(t,'cost_usd')>0)?`${ts.filter(t=>traceMetricValue(t,'cost_usd')!==null).length}/${ts.length} traces reporting cost`:`All ${ts.filter(t=>traceMetricValue(t,'cost_usd')!==null).length} reported costs are $0 (local inference)`]];return `<div class="metrics">${cards.map(([l,v,h])=>`<article class="metric"><div class="metric-label">${escape(l)}</div><div class="value">${escape(v)}</div><small>${escape(h)}</small></article>`).join('')}</div>${visuals()?overviewVisuals(ts,metric):''}<div class="grid">${panel('Run performance',`${metricLabel(metric)||'No metric yet'} · latest score per trace`,runs.length?`<div class="panel-body">${runs.map(r=>`<div class="row"><span class="run-name" title="${escape(r.run)}">${escape(midEllipsis(runLabel(r.run),34))}</span>${visuals()?`<progress class="bar" max="1" value="${r.value||0}" aria-label="Run mean score"></progress>`:''}<span class="score">${pct(r.value)}</span><small class="muted">${r.coverage}</small></div>`).join('')}${unscoredRuns?`<p class="muted">${unscoredRuns} further run${unscoredRuns===1?' has':'s have'} no ${escape(metricLabel(metric)||'score')} values (for example stage or setup runs) and ${unscoredRuns===1?'is':'are'} not listed.</p>`:''}</div>`:empty('No run has scores for this metric yet.'))}${panel('Evaluation coverage','Know what the scores represent',`<div class="panel-body"><div class="row"><span>Successful traces</span><strong>${ts.filter(t=>t.status==='ok').length} / ${ts.length}</strong></div><div class="row"><span>Scored cases</span><strong>${scored.length} / ${ts.length}</strong></div><div class="row"><span>Unscored cases</span><strong>${ts.length-scored.length}</strong></div><div class="callout">Scores describe recorded evidence. Compare matching cases in Improvement tracking before interpreting a change as progress.</div></div>`)}</div>${panel('Recent traces','Inspect the decisions behind the outcome',traceTable(ts.slice(-5).reverse()))}`;},
 scoreboard(ts){return panel('Scoreboard','Each metric uses the latest score per trace. Coverage is shown beside the mean.',metrics().length?(visuals()?`<div class="visual-grid">${metrics().map(m=>panel(metricLabel(m),'Latest scores; unpaired run means',barChart(runScores(ts,m).filter(r=>r.value!==null).map(r=>({label:runLabel(r.run),full:r.run,value:r.value,note:r.coverage+' scored'})),{label:metricLabel(m)+' run scores',max:1,format:pct}))).join('')}</div>`:'')+table(['Run',...metrics().map(metricLabel)], [...new Set(ts.map(t=>t.run_id))].map(run=>`<tr><td><strong>${runCell(run)}</strong></td>${metrics().map(m=>{const r=runScores(ts.filter(t=>t.run_id===run),m)[0];return `<td><strong>${pct(r.value)}</strong> <small class="muted">${r.coverage} scored</small></td>`;}).join('')}</tr>`)):empty('Open a trace and add a score to create a metric.'));},
 traces(ts){return panel('Trace explorer',`${ts.length} traces · select a case to inspect messages, spans, and scoring history`,traceTable(ts));},
 improvements(){const runs=[...new Set(data.traces.map(t=>t.run_id))],ms=metrics();if(runs.length<2||!ms.length)return panel('Improvement tracking','Paired case comparison',empty('Add two runs and score matching cases using the same metric.'));
  const st=compareSelection(runs,ms),opt=(r,sel)=>`<option value="${escape(r)}"${r===sel?' selected':''}>${escape(runLabel(r))}</option>`;
  return panel('Improvement tracking','Uses all workspace runs. One trace per case per run; duplicate trials must use separate runs.',`<form id="compare-form" class="compare-controls"><label>Baseline <select name="baseline" title="${escape(st.baseline)}">${runs.map(r=>opt(r,st.baseline)).join('')}</select></label><label>Candidate <select name="candidate" title="${escape(st.candidate)}">${runs.map(r=>opt(r,st.candidate)).join('')}</select></label><label>Metric <select name="metric">${ms.map(m=>`<option value="${escape(m)}"${m===st.metric?' selected':''}>${escape(metricLabel(m))}</option>`).join('')}</select></label><button type="submit">Compare runs</button></form><div class="comparison">${empty('Comparing…')}</div>`);},
 tools(ts){const tools=Object.create(null);ts.flatMap(t=>t.spans).filter(s=>s.kind==='tool').forEach(s=>{const item=tools[s.name]??={count:0,errors:0,duration:0,timed:0};item.count++;item.errors+=s.status==='error'?1:0;if(Number.isFinite(s.duration_ms)){item.duration+=s.duration_ms;item.timed++;}});return panel('Tool activity','Recorded tool spans only',Object.keys(tools).length?(visuals()?barChart(Object.entries(tools).map(([name,t])=>({label:name,value:t.count})),{label:'Recorded tool calls'}):'')+table(['Tool','Calls','Errors','Mean duration'],Object.entries(tools).map(([name,s])=>`<tr><td>${escape(name)}</td><td>${s.count}</td><td>${s.errors}</td><td>${s.timed?(s.duration/s.timed).toFixed(0)+' ms ('+s.timed+'/'+s.count+' timed)':'Not recorded'}</td></tr>`)):empty('No tool spans recorded in this scope.'));},
 memory(ts){const events=ts.flatMap(t=>t.memory_events.map(e=>({...e,trace_id:t.id})));return panel('Memory events','Explicit producer events; no inferred memory or hidden reasoning',events.length?table(['Trace','Operation','Evidence'],events.map(e=>`<tr><td>${escape(e.trace_id)}</td><td>${escape(e.operation||'event')}</td><td><pre>${escape(JSON.stringify(e,null,2))}</pre></td></tr>`)):empty('No memory events recorded. Producers can emit retrieval, write, and update events.'));}
};
function runOptions(selected){return '<option value="">All runs</option>'+[...new Set(data.traces.map(t=>t.run_id))].map(r=>`<option value="${escape(r)}"${r===selected?' selected':''} title="${escape(r)}">${escape(runLabel(r))}</option>`).join('');}
// Toolbar controls only appear on pages whose sections use them (search: trace lists; scope: run-filtered views; metric: Overview).
function syncToolbar(page){
 const comps=new Set((page?.tabs||[]).flatMap(t=>t.sections.map(x=>x.component)));
 const show=(sel,on)=>{const el=$(sel);if(el)(el.closest?.('label')||el).hidden=!on;};
 const scope=['overview','scoreboard','metrics','traces','tools','memory'].some(c=>comps.has(c));
 show('#search',comps.has('traces'));show('#run-filter',scope);show('#metric-filter',comps.has('overview'));
 if($('#all-tab'))$('#all-tab').hidden=!(scope||comps.has('traces'));
 if(!comps.has('traces')&&$('#search').value)$('#search').value='';
}
const THEME_KEY='evalboard.themeMode',THEME_MODES=['light','system','dark'];
function storedThemeMode(){try{const v=localStorage.getItem(THEME_KEY);return THEME_MODES.includes(v)?v:null;}catch(e){return null;}}
function themeMode(){return storedThemeMode()||(THEME_MODES.includes(data?.layout?.theme)?data.layout.theme:'system');}
const systemDark=window.matchMedia?window.matchMedia('(prefers-color-scheme: dark)'):null;
function isDark(){const m=themeMode();return m==='dark'||(m==='system'&&!!systemDark?.matches);}
function applyTheme(){const dark=isDark(),mode=themeMode();document.body.classList.toggle('dark',dark);document.documentElement.classList.toggle('dark',dark);document.body.dataset.themeMode=mode;document.body.dataset.theme=dark?'dark':'light';document.querySelectorAll('#theme-toggle [data-theme-mode]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.themeMode===mode)));}
function setThemeMode(mode){if(!THEME_MODES.includes(mode))return;try{localStorage.setItem(THEME_KEY,mode);}catch(e){}if(data)render();else applyTheme();}
document.querySelectorAll('#theme-toggle [data-theme-mode]').forEach(b=>b.addEventListener('click',()=>setThemeMode(b.dataset.themeMode)));
const onSystemThemeChange=()=>{if(themeMode()==='system'){if(data)render();else applyTheme();}};
if(systemDark){if(systemDark.addEventListener)systemDark.addEventListener('change',onSystemThemeChange);else if(systemDark.addListener)systemDark.addListener(onSystemThemeChange);}
window.addEventListener('storage',e=>{if(e.key===THEME_KEY)(data?render():applyTheme());});
applyTheme();
function render(){runLabelMap=computeRunLabels(data.traces.map(t=>t.run_id));interpLabelMap=computeRunLabels((data.interp||[]).map(a=>a.run_id));{const sel=$('#run-filter').value;$('#run-filter').innerHTML=runOptions(sel);$('#run-filter').value=sel;}applyTheme();document.body.dataset.density=data.layout.density||'comfortable';document.body.dataset.corners=data.layout.corners||'rounded';document.title=`${data.layout.title} · Evalboard`;renderBrand();const available=pages();if(!available.some(p=>p.id===active))active=available[0]?.id;const page=available.find(p=>p.id===active);$('#title').textContent=page?.title||data.layout.title;$('#nav').innerHTML=available.map(p=>`<button data-module="${escape(p.id)}" class="${p.id===active?'active':''}">${icons[p.id]||'◈'} &nbsp; ${escape(p.title)}</button>`).join('');$('#breadcrumb').textContent=page?.title||'Workspace';$('#trace-count').textContent=`${data.traces.length} traces${data.traces.some(t=>t.synthetic)?' · SYNTHETIC DEMO':''}`;syncToolbar(page);$('#view').innerHTML=page?renderPage(page):empty('No pages enabled. Open Customize to add pages.');autoCompare();}
async function refresh(){const run=$('#run-filter').value,metric=$('#metric-filter').value;data=await api('/api/workspace');runLabelMap=computeRunLabels(data.traces.map(t=>t.run_id));interpLabelMap=computeRunLabels((data.interp||[]).map(a=>a.run_id));$('#run-filter').innerHTML=runOptions(run);$('#run-filter').value=run;$('#metric-filter').innerHTML=metrics().map(m=>`<option value="${escape(m)}">${escape(metricLabel(m))}</option>`).join('');$('#metric-filter').value=metrics().includes(metric)?metric:(defaultMetric()??'');$('#theme-styles').href='/theme.css?v='+Date.now();render();}
function spanTree(spans,parent=null){return spans.filter(s=>(s.parent_id??null)===parent).map(s=>`<div class="span"><strong>${escape(s.name)}</strong> <span class="badge">${escape(s.kind||'span')}</span> <small>${s.duration_ms===undefined?'Duration not recorded':s.duration_ms+' ms'} · ${escape(s.status||'not recorded')}</small><details><summary>Recorded payload</summary><pre>${escape(JSON.stringify(s,null,2))}</pre></details>${spanTree(spans,s.id)}</div>`).join('');}
function openTrace(id,showVisuals=true){selectedTrace=id;selectedTraceVisuals=showVisuals;const t=data.traces.find(t=>t.id===id);$('#trace-detail').innerHTML=`<p><strong>${escape(t.case_id)}</strong> <span class="badge" title="${escape(t.run_id)}">${escape(runLabel(t.run_id))}</span></p><p class="muted">${escape(t.model)} · ${t.adapter?.missing?.includes('duration_ms')?'Duration not recorded':t.duration_ms+' ms'} · ${t.adapter?.missing?.includes('cost_usd')?'Cost not recorded':'$'+t.cost_usd.toFixed(4)}</p><h3>Producer evidence</h3>${t.adapter?`<details open><summary>Harness import evidence</summary><pre>${escape(JSON.stringify(t.adapter,null,2))}</pre></details>`:''}${t.acp?`<details open><summary>Agent Control Plane · ${escape(t.acp.status)} · approval: ${escape(t.acp.approval_status)}</summary><pre>${escape(JSON.stringify(t.acp,null,2))}</pre></details>`:empty('No additional producer evidence recorded.')}<h3>Messages</h3>${t.messages.map(m=>`<details open><summary>${escape(m.role||'message')}</summary><pre>${escape(typeof m.content==='string'?m.content:JSON.stringify(m.content,null,2))}</pre></details>`).join('')||empty('No messages recorded.')}<h3>Execution spans</h3>${showVisuals&&visuals()?barChart(t.spans.filter(s=>Number.isFinite(s.duration_ms)).map(s=>({label:s.name,value:s.duration_ms})),{label:'Recorded span durations · nested spans may overlap',format:v=>v+' ms'}):''}${spanTree(t.spans)||empty('No spans recorded.')}<h3>Score history</h3>${data.scores.filter(s=>s.trace_id===id).map(s=>`<div class="callout"><strong>${escape(s.metric)} · ${pct(s.value)}</strong><br>${escape(s.rationale)}<br><small>${escape(s.scorer)} · ${escape(s.created_at)}</small></div>`).join('')||empty('No scores yet.')}${interpTraceSection(t)}<h3>Add a score</h3><form id="score-form" class="score-form"><label>Metric<input name="metric" required placeholder="task_success"></label><label>Value (0–1)<input name="value" type="number" min="0" max="1" step="any" required></label><label class="wide">Scorer<input name="scorer" required placeholder="human:your-name or evaluator:version"></label><label class="wide">Rationale<textarea name="rationale" required rows="3"></textarea></label><button class="primary" type="submit">Save score</button><p id="score-notice" role="status"></p></form>`;if(!$('#trace-dialog').open)$('#trace-dialog').showModal();}
function compositionSettings(form){
 const pages=JSON.parse(form.elements.pages_json.value||'null'),definitions=JSON.parse(form.elements.metrics_json.value||'[]');
 if(pages!==null&&!Array.isArray(pages))throw Error('Pages must be an array or null.');
 return {brand:{name:form.elements.brand_name.value,tagline:form.elements.brand_tagline.value,badge:form.elements.brand_badge.value,logo:{type:form.elements.logo_type.value,value:form.elements.logo_value.value}},auto_structure:form.elements.auto_structure.checked,pages:pages??layoutDraft.components.map(id=>({id,title:labels[id],tabs:[{id:'results',title:'Results',sections:[{id,title:labels[id],component:id}]}]})),metrics:definitions};
}
$('#logo-file').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>262144)throw Error('Logo must be under 256 KiB.');if(!['image/png','image/jpeg','image/webp'].includes(file.type))throw Error('Use a PNG, JPEG, or WebP logo.');const bytes=new Uint8Array(await file.arrayBuffer());let binary='';for(const byte of bytes)binary+=String.fromCharCode(byte);const form=$('#layout-form');form.elements.logo_type.value='image';form.elements.logo_value.value=`data:${file.type};base64,${btoa(binary)}`;}catch(error){notice(error.message,true);}finally{e.target.value='';}};
function renderOptions(){ $('#component-options').innerHTML=layoutDraft.components.concat(Object.keys(labels).filter(k=>!layoutDraft.components.includes(k))).map(id=>`<div class="component-option"><input type="checkbox" id="component-${id}" data-enable="${id}" ${layoutDraft.components.includes(id)?'checked':''}><label for="component-${id}">${labels[id]}<small class="muted"> · ${escape(data.components[id])}</small></label><button type="button" data-move="${id}" data-direction="-1" aria-label="Move ${labels[id]} up">↑</button><button type="button" data-move="${id}" data-direction="1" aria-label="Move ${labels[id]} down">↓</button></div>`).join('');}
$('#customize').onclick=()=>{layoutDraft=structuredClone(data.layout);const form=$('#layout-form');const brand=layoutDraft.brand||{};form.elements.brand_name.value=brand.name??'evalboard';form.elements.brand_tagline.value=brand.tagline??'Local evaluation studio';form.elements.brand_badge.value=brand.badge??'LAB';form.elements.logo_type.value=brand.logo?.type||'mark';form.elements.logo_value.value=brand.logo?.value||'▧';form.elements.auto_structure.checked=layoutDraft.auto_structure===true;form.elements.pages_json.value=JSON.stringify(data.pages??layoutDraft.pages??null,null,2);form.elements.metrics_json.value=JSON.stringify(layoutDraft.metrics||[],null,2);$('#metric-catalog').textContent=JSON.stringify(data.metric_catalog||[],null,2);form.elements.title.value=layoutDraft.title;form.elements.theme.value=layoutDraft.theme;form.elements.visuals.checked=layoutDraft.visuals!==false;form.elements.density.value=layoutDraft.density||'comfortable';form.elements.corners.value=layoutDraft.corners||'rounded';$('#color-options').innerHTML=['accent','secondary','background','surface','text','muted','positive','negative','line'].map(key=>`<label>${key}<input name="color-${key}" placeholder="Theme default" pattern="#[0-9a-fA-F]{6}" value="${escape(layoutDraft.colors?.[key]||'')}"></label>`).join('');renderOptions();$('#settings').showModal();};
$('#close-settings').onclick=()=>$('#settings').close();$('#close-trace').onclick=()=>$('#trace-dialog').close();
$('#component-options').onchange=e=>{const id=e.target.dataset.enable;if(!id)return;$('#layout-form').elements.auto_structure.checked=false;$('#layout-form').elements.pages_json.value='null';layoutDraft.components=e.target.checked?[...layoutDraft.components,id]:layoutDraft.components.filter(k=>k!==id);renderOptions();};
$('#component-options').onclick=e=>{const id=e.target.dataset.move;if(!id)return;$('#layout-form').elements.auto_structure.checked=false;$('#layout-form').elements.pages_json.value='null';const a=layoutDraft.components,i=a.indexOf(id),j=i+Number(e.target.dataset.direction);if(i>=0&&j>=0&&j<a.length){[a[i],a[j]]=[a[j],a[i]];renderOptions();}};
$('#layout-form').onsubmit=async e=>{e.preventDefault();try{await api('/api/layout',{...layoutDraft,...compositionSettings(e.target),title:e.target.elements.title.value,theme:e.target.elements.theme.value,visuals:e.target.elements.visuals.checked,density:e.target.elements.density.value,corners:e.target.elements.corners.value,colors:Object.fromEntries([...new FormData(e.target)].filter(([k,v])=>k.startsWith('color-')&&v).map(([k,v])=>[k.slice(6),v]))});$('#settings').close();await refresh();notice('Workspace configuration saved.');}catch(err){notice(err.message,true);}};
$('#metric-filter').onchange=render;$('#run-filter').onchange=render;$('#search').oninput=render;$('#all-tab').onclick=()=>{$('#run-filter').value='';$('#search').value='';render();};
document.addEventListener('click',e=>{const module=e.target.closest('[data-module]'),trace=e.target.closest('[data-trace]');if(module){active=module.dataset.module;$('#search').value='';render();}const tab=e.target.closest('[data-tab]');if(tab){activeTabs[active]=tab.dataset.tab;render();}if(trace){const section=currentSection(trace);openTrace(trace.dataset.trace,section?.visuals!==false);}});
document.addEventListener('change',e=>{if(e.target.closest?.('#compare-form'))runCompare(e.target.closest('#compare-form'));});
document.addEventListener('submit',async e=>{if(e.target.id==='score-form'){e.preventDefault();const fields=Object.fromEntries(new FormData(e.target));try{await api('/api/scores',{...fields,trace_id:selectedTrace,value:Number(fields.value)});await refresh();openTrace(selectedTrace,selectedTraceVisuals);}catch(err){$('#score-notice').textContent=err.message;}}if(e.target.id==='compare-form'){e.preventDefault();runCompare(e.target);}});
let pendingImport=null,previewedImport=null;
function importPayload(){const form=$('#import-options'),mapping=form.elements.mapping.value.trim(),namespace=form.elements.namespace.value.trim();return {data:pendingImport,...(mapping?{mapping:JSON.parse(mapping)}:{}),...(namespace?{namespace}:{})};}
async function previewImport(){
 $('#import-confirm').disabled=true;previewedImport=null;$('#import-feedback').textContent='Preparing preview…';
 try{const payload=importPayload(),result=await api('/api/import/preview',payload);if(!$('#import-dialog').open||JSON.stringify(importPayload())!==JSON.stringify(payload)){return;}previewedImport=payload;$('#import-preview').innerHTML=`<div class="callout">${result.report.record_count} traces · ${result.report.score_count} attributed scores · ${result.report.warnings.length} notices</div>`+(result.report.warnings.length?table(['Record (zero-based)','Field','Notice'],result.report.warnings.map(w=>`<tr><td>${w.record}</td><td>${escape(w.field)}</td><td>${escape(w.message)}</td></tr>`)):'')+`<details><summary>First normalized trace</summary><pre>${escape(JSON.stringify(result.traces[0],null,2))}</pre></details>`;$('#import-feedback').textContent='Preview ready. Missing evidence remains marked as unknown.';$('#import-confirm').disabled=false;}catch(error){$('#import-preview').innerHTML='';$('#import-feedback').textContent=error.message;}
}
$('#import-options').onsubmit=e=>{e.preventDefault();previewImport();};
$('#import-options').oninput=()=>{previewedImport=null;$('#import-confirm').disabled=true;$('#import-feedback').textContent='Update the preview to apply mapping changes.';};
$('#close-import').onclick=()=>{$('#import-dialog').close();pendingImport=null;previewedImport=null;};
$('#import-confirm').onclick=async()=>{if(!previewedImport)return;$('#import-confirm').disabled=true;try{const result=await api('/api/import',previewedImport);$('#import-dialog').close();pendingImport=null;previewedImport=null;await refresh();notice(`Imported ${result.ingested} traces and ${result.report.score_count} attributed evaluation records (existing traces are unchanged). ${result.report.warnings.length} import notices were shown in the preview.`);}catch(error){$('#import-feedback').textContent=error.message;$('#import-confirm').disabled=false;}};
$('#import').onclick=()=>$('#file').click();$('#file').onchange=async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>10_000_000)throw Error('File exceeds 10 MB. Split it into smaller batches.');const text=await file.text();try{pendingImport=JSON.parse(text);}catch{pendingImport=text.split(/\r?\n/).filter(l=>l.trim()).map(l=>JSON.parse(l));}$('#import-options').reset();$('#import-preview').innerHTML='';$('#import-dialog').showModal();await previewImport();}catch(error){notice(error.message,true);}finally{e.target.value='';}};
refresh().catch(err=>notice(`Could not load workspace: ${err.message}`,true));
