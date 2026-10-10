(function(){
 const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const num=x=>Number.isFinite(x)?x.toFixed(4):'Unstable';
 let traceState={active:false,index:0,split:'val',stage:0,epoch:0},page=0,runs={churn:[],campaign:[],sales:[]},serial=0;
 const pageSize=12,colors=['#08776c','#6942b7','#ad4c22'];
 const activities={
  depth:{title:'Do hidden layers help?',question:'Can a network with hidden layers model campaign response better than a direct input-to-output model?',dataset:'campaign',a:{hidden:[],selected:[0,1,2],rate:.05},b:{hidden:[4,3],selected:[0,1,2],rate:.05},description:'Run A: no hidden layers. Run B: two hidden layers (4 and 3 neurons). Both use all inputs, a learning rate of 0.05 and 300 epochs. Hidden neurons in Run B use tanh.',prompt:'Compare validation accuracy and loss. Explain why a nonlinear model might help, and why extra complexity need not always improve validation results.'},
  inputs:{title:'Does every input help?',question:'How does removing support-ticket history affect a churn model?',dataset:'churn',a:{hidden:[4,3],selected:[0,1,2],rate:.05},b:{hidden:[4,3],selected:[0,2],rate:.05},description:'Run A: all three inputs. Run B: monthly usage and tenure only. Both use 4 and 3 hidden neurons, tanh, a learning rate of 0.05 and 300 epochs.',prompt:'Explain whether support-ticket history helped in this dataset. Removing a feature also changes the number of weights and initialisation; this is a model comparison, not proof of causation or a definitive feature-importance test.'},
  rate:{title:'How does the learning rate matter?',question:'Does taking larger learning steps improve sales predictions within a fixed training budget?',dataset:'sales',a:{hidden:[4,3],selected:[0,1,2],rate:.005},b:{hidden:[4,3],selected:[0,1,2],rate:.1},description:'Run A: learning rate 0.005. Run B: learning rate 0.1. Both use all inputs, 4 and 3 hidden neurons, tanh and 300 epochs.',prompt:'Compare the learning curves and validation MAE in £k. Which learned faster? Would a larger learning rate always be better?'}
 };
 const notes=Object.fromEntries(Object.keys(activities).map(k=>[k,{hypothesis:'',reflection:'',A:null,B:null,configured:null,finished:false}]));
 const key=()=>$('dataset').value,activityKey=()=>$('activitySelect').value;
 const settings=()=>({dataset:key(),selected:[...data.selected],names:data.selected.map(j=>data.spec.names[j]),hidden:[...hidden],activation:net.activation,rate:+$('rate').value,type:data.spec.type});
 const example=()=>data[traceState.split][traceState.index]||data.val[0];
 const exampleLabel=()=>`${traceState.split==='val'?'validation':'training'} example ${traceState.index+1}`;
 const stepList=()=>LabCore.trace(net,data,example());
 function traceView(){if(!traceState.active)return null;const s=stepList()[traceState.stage];return {layer:s.layer,revealed:s.kind==='sum'?s.layer-1:s.layer}}
 function populateExamples(){
  $('traceSplit').value=traceState.split;
  $('traceExample').innerHTML=data[traceState.split].map((row,i)=>`<option value="${i}">${traceState.split==='val'?'V':'T'}${String(i+1).padStart(3,'0')} · actual ${data.spec.type==='regression'?'£'+row.y+'k':row.y?'yes':'no'}</option>`).join('');
  $('traceExample').value=String(traceState.index);
 }
 function endTrace(){traceState.active=false;$('traceWalkthrough').hidden=true;updateTraceNotice()}
 function updateTraceNotice(){
  $('traceEpoch').textContent='Current weights · epoch '+epoch;
  $('traceNotice').textContent=epoch===0?'These are untrained, random weights. Train first to trace a learned prediction, or explore the initial calculation.':`Using ${exampleLabel()} with weights from epoch ${epoch}. Tracing changes no weights and uses no target value as an input.`;
 }
 function startTrace(){stop();traceState.active=true;traceState.stage=0;traceState.epoch=epoch;$('showValues').checked=true;drawTrace();network()}
 function drawTrace(){
  updateTraceNotice();$('traceWalkthrough').hidden=!traceState.active;if(!traceState.active)return;
  const steps=stepList(),s=steps[traceState.stage];
  $('traceProgress').textContent=`Stage ${traceState.stage+1} of ${steps.length} · ${exampleLabel()} · epoch ${epoch}`;
  $('traceStageTitle').textContent=s.title;
  $('tracePrev').disabled=traceState.stage===0;$('traceNext').disabled=traceState.stage===steps.length-1;
  let body='';
  if(s.kind==='input'){
   body='<p>Subtract each training mean and divide by its training standard deviation. The target is excluded.</p><div class="table-scroll"><table><thead><tr><th>Input</th><th>Original</th><th>Mean</th><th>Std. deviation</th><th>Scaled input</th></tr></thead><tbody>'+s.values.map(v=>`<tr><td>${v.name}</td><td>${v.raw}</td><td>${num(v.mean)}</td><td>${num(v.sd)}</td><td><b>${num(v.value)}</b></td></tr>`).join('')+'</tbody></table></div>';
  }else if(s.kind==='sum'){
   body='<p>For each neuron, multiply incoming activations by their weights, add the contributions, then add the bias. Values are rounded for display.</p><div class="trace-calculations">'+s.neurons.map((n,j)=>`<article><h4>${neuronName(s.layer,j)}</h4><p class="trace-formula">${n.terms.map(t=>`(${num(t.activation)} × ${num(t.weight)})`).join(' + ')}</p><p>Weighted sum <b>${num(n.sum)}</b> + bias <b>${num(n.bias)}</b> = <b>${num(n.z)}</b></p></article>`).join('')+'</div>';
  }else if(s.kind==='activation'){
   const rule={Tanh:'tanh(z), between −1 and 1',ReLU:'max(0, z)',Sigmoid:'1 ÷ (1 + e⁻ᶻ), between 0 and 1',Linear:'z, unchanged'}[s.fn];
   body=`<p>Apply <strong>${s.fn}</strong>: ${rule}. ${s.layer===net.sizes.length-1?'This is the output activation.':'These activations are passed to the next layer.'}</p><div class="table-scroll"><table><thead><tr><th>Neuron</th><th>Weighted sum + bias (z)</th><th>Activation</th></tr></thead><tbody>${s.neurons.map((n,j)=>`<tr><td>${neuronName(s.layer,j)}</td><td>${num(n.z)}</td><td><b>${num(n.value)}</b></td></tr>`).join('')}</tbody></table></div>`;
  }else{
   body=data.spec.type==='regression'?`<p>Convert the standardised output back to £k using the training target statistics.</p><p class="trace-formula">${num(s.raw)} × ${num(data.ys)} + ${num(data.ym)} = <strong>£${s.value.toFixed(2)}k</strong></p><p>Actual sales: £${s.actual}k. Absolute error on this example: £${Math.abs(s.value-s.actual).toFixed(2)}k.</p>`:`<p class="trace-answer">${(s.value*100).toFixed(2)}% probability of “yes”</p><p>At a 50% threshold, the prediction is <strong>${s.value>=.5?'Yes':'No'}</strong>. The actual answer is <strong>${s.actual?'Yes':'No'}</strong>.</p>`;
   body+='<p>The actual answer is used only for comparison here. This forward pass did not perform backpropagation or update any parameter.</p>';
  }
  $('traceStageBody').innerHTML=body;
 }
 function selectExample(split,index,start=false){stop();traceState.split=split;traceState.index=index;endTrace();populateExamples();if(start)startTrace();else network()}
 $('traceSplit').onchange=()=>selectExample($('traceSplit').value,0);
 $('traceExample').onchange=()=>selectExample(traceState.split,+$('traceExample').value);
 $('traceStart').onclick=startTrace;$('traceEnd').onclick=()=>{endTrace();network()};
 $('tracePrev').onclick=()=>{traceState.stage=Math.max(0,traceState.stage-1);drawTrace();network()};
 $('traceNext').onclick=()=>{traceState.stage=Math.min(stepList().length-1,traceState.stage+1);drawTrace();network()};
 function drawData(){
  $('downloadCsv').href='data:text/csv;charset=utf-8,'+encodeURIComponent(LabCore.csv(data));$('downloadCsv').download=`neural-lab-${key()}-240-examples.csv`;
  const filter=$('dataFilter').value,rows=LabCore.rows(data).filter(d=>filter==='all'||d.split===filter),pages=Math.ceil(rows.length/pageSize);page=Math.min(page,pages-1);
  $('dataTable').innerHTML='<thead><tr><th>Example</th><th>Split</th>'+data.spec.names.map((n,j)=>`<th>${n}${data.selected.includes(j)?'':' (not selected)'}</th>`).join('')+`<th>${data.spec.target}</th><th>Explore</th></tr></thead><tbody>`+rows.slice(page*pageSize,(page+1)*pageSize).map(d=>`<tr><td>${d.id}</td><td><span class="split-badge ${d.split}">${d.split==='train'?'Training':'Validation'}</span></td>${d.row.x.map(v=>`<td>${v}</td>`).join('')}<td><b>${data.spec.type==='regression'?'£'+d.row.y+'k':d.row.y?'Yes':'No'}</b></td><td><button class="secondary trace-row" type="button" data-split="${d.split}" data-index="${d.index}" aria-label="Trace ${d.id}">Trace</button></td></tr>`).join('')+'</tbody>';
  $('dataPageInfo').textContent=`${page*pageSize+1}–${Math.min((page+1)*pageSize,rows.length)} of ${rows.length} · page ${page+1} of ${pages}`;
  $('dataPrev').disabled=page===0;$('dataNext').disabled=page===pages-1;
 }
 $('dataFilter').onchange=()=>{page=0;drawData()};$('dataPrev').onclick=()=>{page--;drawData()};$('dataNext').onclick=()=>{page++;drawData()};
 $('dataTable').addEventListener('click',e=>{const button=e.target.closest('.trace-row');if(!button)return;selectExample(button.dataset.split,+button.dataset.index,true);$('tracePanel').scrollIntoView({behavior:'smooth',block:'start'});$('traceStart').focus({preventScroll:true})});
 
 function captureRun(label){
  stop();const group=runs[key()],fingerprint=JSON.stringify([net.w,net.b]);const existing=group.find(r=>r.fingerprint===fingerprint&&r.epoch===epoch);if(existing)return existing;
  if(group.length>=3)throw Error('Three runs are already captured for this question. Remove one comparison card to make space.');
  const r=LabCore.capture(net,data,history,epoch,settings());r.id=++serial;r.label=label||'Run '+r.id;r.fingerprint=fingerprint;group.push(r);drawComparisons();return r;
 }
 $('captureRun').onclick=()=>{try{const r=captureRun();$('compareStatus').textContent=`${r.label} captured at epoch ${r.epoch}. Change one setting, retrain, then capture another run.`}catch(e){$('compareStatus').textContent=e.message}};
 function compareChart(group){
  const maxEpoch=Math.max(1,...group.map(r=>r.epoch)),maxLoss=Math.max(.01,...group.flatMap(r=>r.history.flatMap(h=>[h.t,h.v])))*1.1;
  let svg='';for(let i=0;i<5;i++){const y=20+i*48;svg+=S('line',`x1="55" y1="${y}" x2="780" y2="${y}" stroke="#c8dbe7"`)+text(0,y+4,(maxLoss*(1-i/4)).toFixed(2))}
  group.forEach((r,i)=>{for(const k of ['t','v'])svg+=S('polyline',`points="${r.history.map(h=>`${55+h.e/maxEpoch*720},${212-h[k]/maxLoss*192}`).join(' ')}" fill="none" stroke="${colors[i]}" stroke-width="2.5" ${k==='t'?'stroke-dasharray="5 4"':''}`)});
  svg+=text(55,243,'0')+text(625,243,maxEpoch+' epochs')+text(4,12,'Loss');$('compareChart').innerHTML=svg;
 }
 function drawComparisons(){
  const group=runs[key()];$('compareStatus').textContent=group.length?`${group.length} of 3 snapshots for ${data.spec.output}. Other business questions have separate comparisons.`:'No captured runs for this business question yet. Train, then capture a snapshot.';
  $('compareCards').innerHTML=group.map((r,i)=>`<article class="comparison-card" style="--run-color:${colors[i]}"><div class="panel-head"><h3>${esc(r.label)}</h3><button class="remove-run" type="button" data-remove="${r.id}" aria-label="Remove ${esc(r.label)}">×</button></div><p><strong>Epoch ${r.epoch}</strong> · ${r.parameters} parameters</p><dl><dt>Inputs</dt><dd>${r.settings.names.join(', ')}</dd><dt>Hidden layers</dt><dd>${r.settings.hidden.length?r.settings.hidden.join(' → ')+' neurons':'None'}</dd><dt>Activation / learning rate</dt><dd>${r.settings.hidden.length?r.settings.activation:'No hidden activation'} / ${r.settings.rate}</dd><dt>Training / validation loss</dt><dd>${num(r.trainLoss)} / <b>${num(r.valLoss)}</b></dd><dt>${r.settings.type==='regression'?'Validation MAE':'Validation accuracy'}</dt><dd class="comparison-score">${r.settings.type==='regression'?'£'+r.mae.toFixed(2)+'k':(r.accuracy*100).toFixed(1)+'%'}</dd></dl></article>`).join('');
  $('compareChartWrap').hidden=!group.length;if(group.length)compareChart(group);
  $('captureRun').disabled=epoch<1||group.length>=3;
 }
 $('compareCards').onclick=e=>{const b=e.target.closest('[data-remove]');if(!b)return;runs[key()]=runs[key()].filter(r=>r.id!==+b.dataset.remove);drawComparisons()};
 function configMatches(c,a){return key()===a.dataset&&JSON.stringify(hidden)===JSON.stringify(c.hidden)&&JSON.stringify(data.selected)===JSON.stringify(c.selected)&&net.activation==='tanh'&&+$('rate').value===c.rate}
 function currentActivitySide(){const a=activities[activityKey()];return configMatches(a.a,a)?'A':configMatches(a.b,a)?'B':null}
 function drawActivity(){
  const a=activities[activityKey()],n=notes[activityKey()];$('activityBrief').innerHTML=`<h3>${a.question}</h3><p>${a.description}</p><p class="help">${a.prompt}</p>`;
  $('activityHypothesis').value=n.hypothesis;$('activityReflection').value=n.reflection;drawEvidence();activityControls();
 }
 function drawEvidence(){
  const n=notes[activityKey()];$('activityEvidence').innerHTML=['A','B'].map(side=>{const r=n[side];return `<div class="activity-evidence"><b>Run ${side}</b><p>${r?`Epoch ${r.epoch} · validation loss ${num(r.valLoss)} · ${r.settings.type==='regression'?'MAE £'+r.mae.toFixed(2)+'k':'accuracy '+(r.accuracy*100).toFixed(1)+'%'}`:'Not recorded yet'}</p></div>`}).join('');
  $('activityFeedback').textContent=n.finished?'Reflection complete. Your notes and both results remain here until this page is reloaded.':'';
 }
 function activityControls(){
  const n=notes[activityKey()],side=currentActivitySide();$('activityA').disabled=!n.hypothesis.trim();$('activityB').disabled=!n.hypothesis.trim();$('activityTrain').disabled=!side||epoch>=300||!n.hypothesis.trim();$('activityCapture').disabled=!side||epoch!==300||!n.hypothesis.trim();
  $('activityStatus').textContent=!n.hypothesis.trim()?'Start by writing your prediction.':!side?'Choose Set up Run A or Set up Run B.':`Current settings match Run ${side}. Epoch ${epoch} of 300.${epoch>300?' Set up this run again to compare at exactly 300 epochs.':''}`;
 }
 function applyActivity(side){
  stop();const a=activities[activityKey()],c=side==='A'?a.a:a.b,n=notes[activityKey()];if(!n.hypothesis.trim())return;
  $('dataset').value=a.dataset;inputs();$('inputs').querySelectorAll('input').forEach(el=>el.checked=c.selected.includes(+el.value));hidden=[...c.hidden];$('layers').value=String(hidden.length);layers();$('activation').value='tanh';$('rate').value=String(c.rate);$('rateValue').textContent=c.rate.toFixed(3);reset();n.configured=side;activityControls();
 }
 $('activitySelect').onchange=drawActivity;
 $('activityHypothesis').oninput=()=>{notes[activityKey()].hypothesis=$('activityHypothesis').value;notes[activityKey()].finished=false;activityControls();$('activityFeedback').textContent=''};
 $('activityReflection').oninput=()=>{notes[activityKey()].reflection=$('activityReflection').value;notes[activityKey()].finished=false;$('activityFeedback').textContent=''};
 $('activityA').onclick=()=>applyActivity('A');$('activityB').onclick=()=>applyActivity('B');
 $('activityTrain').onclick=()=>{if(!currentActivitySide()||epoch>=300)return;stop();endTrace();$('train').textContent='Ⅱ Pause training';$('status').textContent='Training activity to epoch 300';timer=setInterval(()=>{advance(Math.min(2,300-epoch));if(epoch>=300)stop()},45)};
 $('activityCapture').onclick=()=>{const side=currentActivitySide();if(!side||epoch!==300)return;try{const r=captureRun(activities[activityKey()].title+' · '+side);notes[activityKey()][side]=JSON.parse(JSON.stringify(r));notes[activityKey()].finished=false;drawEvidence();$('activityStatus').textContent=`Run ${side} recorded and added to the comparison panel.`}catch(e){$('activityStatus').textContent=e.message}};
 $('activityFinish').onclick=()=>{const n=notes[activityKey()];if(!n.A||!n.B||!n.reflection.trim()){$('activityFeedback').textContent='Record both runs and write your explanation before completing the activity.';return}n.finished=true;drawEvidence()};
 function resetLab(){traceState={active:false,index:0,split:'val',stage:0,epoch:0};page=0;$('traceWalkthrough').hidden=true;populateExamples();drawComparisons();activityControls();updateTraceNotice()}
 function trainingChanged(){if(traceState.active&&traceState.epoch!==epoch)endTrace();updateTraceNotice();$('captureRun').disabled=epoch<1||runs[key()].length>=3;activityControls()}
 window.Lab={example,exampleLabel,traceView,endTrace,reset:resetLab,drawData,trainingChanged};
 populateExamples();drawData();drawComparisons();drawActivity();updateTraceNotice();network();
})();
