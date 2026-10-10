const $=id=>document.getElementById(id);let data,net,epoch=0,history=[],timer=null,hidden=[4,3],inspection=null;const fmt=x=>x.toFixed(3);const S=(tag,attrs,txt='')=>`<${tag} ${attrs}>${txt}</${tag}>`;const text=(x,y,t,color='#48617c',size=11)=>S('text',`x="${x}" y="${y}" fill="${color}" font-size="${size}" font-family="system-ui"`,t);
function inputs(){const s=Neural.specs[$('dataset').value];$('inputs').innerHTML=s.names.map((n,i)=>`<label class="check"><input type="checkbox" value="${i}" checked>${n}</label>`).join('');$('inputs').querySelectorAll('input').forEach(el=>el.onchange=()=>{if(!$('inputs').querySelector('input:checked')){el.checked=true;return}reset()});$('description').textContent=s.question;$('outputLabel').textContent=s.output;$('outputHelp').textContent=s.type==='regression'?'Linear output · a number, not a probability.':'Sigmoid output · a probability from 0 to 1. Values ≥ 0.5 predict “yes”.'}
function layers(){const n=+$('layers').value;hidden=Array.from({length:n},(_,i)=>hidden[i]||4);$('layerControls').innerHTML=hidden.map((n,i)=>`<div class="neuron-row"><label for="neurons${i}">Layer ${i+1} neurons</label><select id="neurons${i}" class="small">${[1,2,3,4,5,6,7,8].map(v=>`<option ${v===n?'selected':''}>${v}</option>`).join('')}</select></div>`).join('');hidden.forEach((_,i)=>$('neurons'+i).onchange=e=>{hidden[i]=+e.target.value;reset()});$('activation').disabled=n===0}
function stop(){clearInterval(timer);timer=null;$('train').textContent=epoch>=1000?'Training complete':'▶ Train network';$('train').disabled=epoch>=1000;$('step').disabled=epoch>=1000;$('status').textContent=epoch>=1000?'1000 epochs complete':epoch?'Paused':'Ready to train'}
function reset(){stop();epoch=0;inspection=null;const selected=[...$('inputs').querySelectorAll('input:checked')].map(e=>+e.value);data=Neural.dataset($('dataset').value,selected);net=new Neural.Network([selected.length,...hidden,1],$('activation').value,data.spec.type);history=[];$('activationHelp').textContent={tanh:'Allows positive and negative signals; useful for these standardised inputs.',relu:'Outputs zero for negative values; otherwise passes the value through.',sigmoid:'Compresses signals to 0–1. Extreme values can make learning slow.'}[$('activation').value];$('train').disabled=false;$('step').disabled=false;$('status').textContent='Ready to train';$('train').textContent='▶ Train network';window.Lab?.reset();drawTable();setupInference();render()}
function drawTable(){if(window.Lab){window.Lab.drawData();return}$('dataTable').innerHTML='<thead><tr><th>Example</th>'+data.spec.names.map(n=>`<th>${n}</th>`).join('')+`<th>${data.spec.target}</th></tr></thead><tbody>`+data.train.slice(0,6).map((d,i)=>`<tr><td>0${i+1}</td>${d.x.map(v=>`<td>${v}</td>`).join('')}<td><b>${data.spec.type==='regression'?'£'+d.y+'k':d.y?'Yes':'No'}</b></td></tr>`).join('')+'</tbody>'}
function displayedExample(){return window.Lab?.example()||data.val[0]}
function neuronName(l,j){return l===0?'Input '+(j+1):l===net.sizes.length-1?'Output':'Hidden '+l+', neuron '+(j+1)}
function detailedNumber(v){return Number.isFinite(v)?v.toFixed(6):String(v)}
function edgeLayout(sizes){
 for(let attempt=0;attempt<6;attempt++){
 const gap=300+attempt*80,spacing=70+attempt*12,width=Math.max(800,(sizes.length-1)*gap+150),height=Math.max(370,Math.max(...sizes)*spacing+100);
 const xs=sizes.map((_,i)=>75+i*(width-150)/(sizes.length-1)),ys=sizes.map(n=>Array.from({length:n},(_,j)=>height/2+(j-(n-1)/2)*spacing));
 const labels=new Map();let failed=false;
 for(let l=0;l<sizes.length-1;l++){const used=[];for(let j=0;j<sizes[l+1];j++)for(let k=0;k<sizes[l];k++){
 let pos;const candidates=Array.from({length:71},(_,i)=>.15+i*.01).sort((a,b)=>Math.abs(a-(.25+j*.06))-Math.abs(b-(.25+j*.06)));
 for(const f of candidates){const x=xs[l]+(xs[l+1]-xs[l])*f,y=ys[l][k]+(ys[l+1][j]-ys[l][k])*f;if(used.every(p=>Math.abs(p.x-x)>=52||Math.abs(p.y-y)>=28)){pos={x,y};break}}
 if(!pos){failed=true;break}used.push(pos);labels.set(`w-${l+1}-${j}-${k}`,pos);
 }}
 if(!failed)return {width,height,xs,ys,labels,attempt};
 }throw Error('Cannot place edge labels');
}

let edgeLayoutCache=null;
function network(){
 const show=$('showValues').checked,sizes=net.sizes,acts=net.forward(displayedExample().input);
 const key=sizes.join(',');
 if(show&&edgeLayoutCache?.key!==key)edgeLayoutCache={key,...edgeLayout(sizes)};
 const width=show?edgeLayoutCache.width:Math.max(800,(sizes.length-1)*220+150),height=show?edgeLayoutCache.height:310;
 const xs=show?edgeLayoutCache.xs:sizes.map((_,i)=>75+i*(width-150)/(sizes.length-1));
 const ys=show?edgeLayoutCache.ys:sizes.map(n=>Array.from({length:n},(_,j)=>height/2+(j-(n-1)/2)*31));
 if(show&&!inspection)inspection={type:'neuron',l:sizes.length-1,j:0};
 const selected=inspection,focusLayer=selected?.l,focusNeuron=selected?.j;
 $('valuesHint').hidden=!show;
 $('exampleContext').textContent='Showing '+(window.Lab?.exampleLabel()||'validation example 1')+' · '+data.selected.map(j=>data.spec.names[j]+': '+displayedExample().x[j]).join(' · ');
 const svgEl=$('network');svgEl.setAttribute('viewBox',`0 0 ${width} ${height}`);svgEl.style.minWidth=show&&(!$('allWeights')||$('allWeights').checked)?width+'px':'';
 let svg='',labels='';
 net.w.forEach((mat,l)=>mat.forEach((row,j)=>row.forEach((w,k)=>{
  const chosen=show&&focusLayer===l+1&&focusNeuron===j&&(selected.type==='neuron'||selected.k===k);
  const id=`w-${l+1}-${j}-${k}`,color=w>=0?'#08776c':'#6942b7';
  const line=`x1="${xs[l]}" y1="${ys[l][k]}" x2="${xs[l+1]}" y2="${ys[l+1][j]}"`;
  svg+=S('g',`data-inspect="${id}" tabindex="${show?0:-1}" role="button" aria-label="Weight from ${neuronName(l,k)} to ${neuronName(l+1,j)}: ${detailedNumber(w)}"`,S('line',`${line} stroke="transparent" stroke-width="12"`)+S('line',`${line} stroke="${color}" stroke-opacity="${show?(chosen?1:.5):.15+Math.min(Math.abs(w)*.3,.65)}" stroke-width="${chosen?3:.4+Math.min(Math.abs(w)*1.3,4)}"`));
  if(show&&(!$('allWeights')||$('allWeights').checked||chosen)){const {x,y}=edgeLayoutCache.labels.get(id);
   labels+=S('g',`class="weight-label" data-inspect="${id}"`,S('rect',`x="${x-23}" y="${y-11}" width="46" height="22" rx="5" fill="#f2faff" stroke="${color}"`)+S('text',`x="${x}" y="${y+4}" text-anchor="middle" fill="${color}" font-size="12"`,w.toFixed(2)));
  }
 })));
 svg+=labels;
 sizes.forEach((n,l)=>{
  svg+=text(xs[l]-30,25,l===0?'INPUT':l===sizes.length-1?'OUTPUT':'HIDDEN '+l,'#38627c',12);
  ys[l].forEach((y,j)=>{
   const a=acts[l][j],trace=window.Lab?.traceView(),revealed=!trace||l<=trace.revealed,chosen=trace?l===trace.layer:show&&focusLayer===l&&focusNeuron===j,r=show?24:11;
   svg+=S('g',`data-inspect="n-${l}-${j}" tabindex="${show?0:-1}" role="button" aria-label="${neuronName(l,j)}: ${l===0?'scaled input':'activation'} ${revealed?detailedNumber(a):'not yet revealed'}"`,S('circle',`cx="${xs[l]}" cy="${y}" r="${r}" fill="${!revealed?'#dce8ee':a>=0?'#bce4d6':'#dcd0f3'}" stroke="${chosen?'#c54c34':a>=0?'#08776c':'#6942b7'}" stroke-width="${chosen?3:2}"`)+(show?S('text',`x="${xs[l]}" y="${y+4}" text-anchor="middle" fill="#173e58" font-size="12" pointer-events="none"`,revealed?a.toFixed(2):'?'):''));
   if(!show&&l===0)svg+=text(xs[l]+17,y+4,'x'+(j+1),'#38627c',12);
   if(!show&&l===sizes.length-1)svg+=text(xs[l]-18,y+35,data.spec.type==='regression'?(a*data.ys+data.ym).toFixed(1)+'k':(a*100).toFixed(0)+'%','#08776c',16);
  });
  svg+=text(xs[l]-25,height-20,`${n} ${n===1?'neuron':'neurons'}`,'#38627c',12);
 });
 const focused=document.activeElement?.getAttribute('data-inspect');svgEl.innerHTML=svg;
 if(focused)svgEl.querySelector(`[data-inspect="${focused}"]`)?.focus({preventScroll:true});
 $('networkInspector').hidden=!show||!!window.Lab?.traceView();if(show&&!window.Lab?.traceView())drawInspector(acts);
}
function drawInspector(acts){
 const {l,j,k,type}=inspection,box=$('networkInspector');
 if(l===0){const raw=displayedExample().x[data.selected[j]];
  box.innerHTML=`<h3>${neuronName(l,j)} · ${data.spec.names[data.selected[j]]}</h3><p>Original value: <b>${raw}</b>. Standardised input: <b>${detailedNumber(acts[0][j])}</b>.</p><p class="calculation">(${raw} − ${detailedNumber(data.means[j])}) ÷ ${detailedNumber(data.sd[j])} = ${detailedNumber(acts[0][j])}</p><p>Input neurons carry scaled data; they have no learned bias or activation function.</p>`;return;
 }
 const weights=net.w[l-1][j],bias=net.b[l-1][j],sum=weights.reduce((s,w,i)=>s+w*acts[l-1][i],0),z=sum+bias,a=acts[l][j];
 const fn=l===net.sizes.length-1?(data.spec.type==='regression'?'Linear':'Sigmoid'):net.activation==='relu'?'ReLU':net.activation==='tanh'?'Tanh':'Sigmoid';
 const rows=weights.map((w,i)=>`<tr${type==='weight'&&i===k?' class="selected-weight"':''}><td>${neuronName(l-1,i)}</td><td>${detailedNumber(acts[l-1][i])}</td><td>${detailedNumber(w)}</td><td>${detailedNumber(w*acts[l-1][i])}</td></tr>`).join('');
 box.innerHTML=`<h3>${type==='weight'?`Connection: ${neuronName(l-1,k)} to `:''}${neuronName(l,j)} · Epoch ${epoch}</h3>${type==='weight'?`<p>Selected weight: <b>${detailedNumber(weights[k])}</b>. Its contribution is input activation × weight = <b>${detailedNumber(acts[l-1][k]*weights[k])}</b>.</p>`:''}<div class="table-scroll"><table><thead><tr><th>From</th><th>Activation</th><th>Weight</th><th>Contribution</th></tr></thead><tbody>${rows}</tbody></table></div><p class="calculation">Weighted sum ${detailedNumber(sum)} + bias ${detailedNumber(bias)} = ${detailedNumber(z)}<br>${fn}(${detailedNumber(z)}) = <b>${detailedNumber(a)}</b></p><p>${l===net.sizes.length-1?(data.spec.type==='regression'?`The output is standardised. Sales = (${detailedNumber(a)} × ${detailedNumber(data.ys)}) + ${detailedNumber(data.ym)} = £${(a*data.ys+data.ym).toFixed(2)}k.`:`Predicted probability: ${(a*100).toFixed(2)}%.`):'This activation is passed to the next layer.'} Display rounded to 2 decimals; details to 6. Calculations use full precision.</p>`;
}
function selectNetworkItem(target){
 const id=target.closest('[data-inspect]')?.getAttribute('data-inspect');if(!id)return;
 const [type,l,j,k]=id.split('-');inspection={type:type==='w'?'weight':'neuron',l:+l,j:+j,k:k===undefined?undefined:+k};
 stop();window.Lab?.endTrace();window.Workspace?.show('train');$('showValues').checked=true;network();
}
$('showValues').onchange=()=>network();
$('network').onclick=e=>selectNetworkItem(e.target);
$('network').onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectNetworkItem(e.target)}};
$('network').onfocusin=()=>{if(timer)stop()};
function lossChart(){const w=450,h=190,left=43,top=13,bottom=163,max=Math.max(.05,...history.flatMap(d=>[d.t,d.v]))*1.1;let svg='';for(let i=0;i<4;i++){const y=top+i*(bottom-top)/3;svg+=S('line',`x1="${left}" y1="${y}" x2="436" y2="${y}" stroke="#e7edf2"`)+text(0,y+4,(max*(1-i/3)).toFixed(2))}const end=Math.max(20,epoch);for(const [key,color]of[['t','#08776c'],['v','#6942b7']]){const pts=history.map(d=>`${left+d.e/end*389},${bottom-d[key]/max*(bottom-top)}`).join(' ');svg+=S('polyline',`points="${pts}" fill="none" stroke="${color}" stroke-width="2.5"`)}svg+=text(left,185,'0')+text(360,185,end+' epochs');$('lossChart').innerHTML=svg}
function predictions(v){let svg='';const reg=data.spec.type==='regression';if(reg){const values=data.val.map(d=>d.y),preds=v.preds.map(p=>p*data.ys+data.ym),min=Math.min(0,...values,...preds),max=Math.max(...values,...preds)*1.08,mapX=v=>45+(v-min)/(max-min)*365,mapY=v=>155-(v-min)/(max-min)*135;svg+=S('line',`x1="45" y1="155" x2="410" y2="20" stroke="#9baabc" stroke-dasharray="4 4"`);values.forEach((actual,i)=>svg+=S('circle',`cx="${mapX(actual)}" cy="${mapY(preds[i])}" r="4" fill="#26a68c" fill-opacity=".75"`,S('title','',`Actual £${actual}k; predicted £${preds[i].toFixed(1)}k`)));svg+=text(130,183,'Actual weekly sales (£k)')+text(3,12,'Predicted (£k)');$('predHelp').textContent='Each dot is a store-week. Closer to the dashed perfect-prediction line is better.'}else{svg+=S('line','x1="45" x2="423" y1="90" y2="90" stroke="#b1bcca" stroke-dasharray="4 4"')+text(0,94,'50%');data.val.forEach((d,i)=>{const x=48+i*7.8,y=155-v.preds[i]*130;svg+=S('circle',`cx="${x}" cy="${y}" r="3.7" fill="${d.y?'#08776c':'#6942b7'}"`,S('title','',`Actual ${d.y?'yes':'no'}, predicted probability ${(v.preds[i]*100).toFixed(1)}%`))});svg+=text(3,28,'100%')+text(10,159,'0%')+text(140,183,'Validation examples');$('predHelp').textContent='Actual answers: teal = yes, purple = no. Dots show predicted “yes” probability; the dashed line is the decision threshold.'}$('predChart').innerHTML=svg}
function render(){window.Lab?.trainingChanged();updateInference();const t=net.evaluate(data.train),v=net.evaluate(data.val);if(!Number.isFinite(t.loss)||!Number.isFinite(v.loss)){stop();$('status').textContent='Learning became unstable';$('insight').textContent='Reduce the learning rate and reset the experiment.';return}history.push({e:epoch,t:t.loss,v:v.loss});$('epoch').textContent=String(epoch).padStart(4,'0');$('trainLoss').textContent=fmt(t.loss);$('valLoss').textContent=fmt(v.loss);if(data.spec.type==='regression'){$('scoreLabel').textContent='Validation error (MAE)';$('score').textContent='£'+(v.mae*data.ys).toFixed(1)+'k';const base=data.val.reduce((s,d)=>s+Math.abs(d.y-data.ym),0)/data.val.length;$('baseline').textContent='Mean-only baseline: £'+base.toFixed(1)+'k'}else{$('scoreLabel').textContent='Validation accuracy';$('score').textContent=(v.accuracy*100).toFixed(0)+'%';const majority=data.train.reduce((s,d)=>s+d.y,0)>=96?1:0;const base=data.val.filter(d=>d.y===majority).length/48;$('baseline').textContent='Majority-class baseline: '+(base*100).toFixed(0)+'%'}network();lossChart();predictions(v);if(epoch===0){$('insightTitle').textContent='Start with a prediction.';$('insight').textContent='The connections begin with random weights. Press Train and see whether the network discovers a useful pattern.'}else{const start=history[0].t;$('insightTitle').textContent=t.loss<start?'The network is finding a pattern.':'Learning takes experimentation.';$('insight').textContent=v.loss>t.loss*1.6&&epoch>100?'Training loss is lower than validation loss. The network may be fitting details that do not generalise. Try fewer neurons and compare the gap.':data.spec.type==='regression'?'Loss measures half the mean squared error on standardised sales targets. MAE translates prediction error back into £k, so you can judge its business meaning.':'Accuracy uses a 50% threshold. Loss also reflects confidence: a confident wrong answer is penalised more. Compare validation accuracy with the simple baseline.'}}
function advance(count=1){window.Lab?.endTrace();for(let i=0;i<count&&epoch<1000;i++){net.epoch(data.train,+$('rate').value);epoch++}render();if(epoch>=1000)stop()}

function setupInference(){
 $('inferenceInputs').innerHTML=data.selected.map((j,k)=>{
  const min=Math.min(...data.train.map(d=>d.x[j])),max=Math.max(...data.train.map(d=>d.x[j]));
  return `<div><label for="newValue${k}">${data.spec.names[j]}</label><input id="newValue${k}" name="value${k}" type="number" step="any" min="0" required placeholder="e.g. ${data.train[0].x[j]}" aria-describedby="newRange${k}"><p class="help" id="newRange${k}">Training range: ${min}–${max}</p></div>`;
 }).join('');
 $('inferenceResult').hidden=true;
 $('inferenceForm').oninput=()=>{$('inferenceResult').hidden=true};
}
function updateInference(){
 $('predictButton').disabled=epoch<1;
 $('inferenceState').textContent=epoch<1?'Train at least one epoch to make a prediction.':`Ready to predict using epoch ${epoch}. Predict pauses any ongoing training.`;
 if(!$('inferenceResult').hidden&&$('inferenceResult').dataset.epoch!==String(epoch)){
  $('inferenceResult').hidden=true;
 }
}
function predictExample(){
 if(epoch<1)return;
 const form=$('inferenceForm');if(!form.reportValidity())return;
 const values=data.selected.map((_,k)=>$('newValue'+k).valueAsNumber);
 stop();const result=$('inferenceResult');result.hidden=false;result.dataset.epoch=String(epoch);
 try{
  const {value}=Neural.predict(net,data,values);
  const outside=data.selected.filter((j,k)=>values[k]<Math.min(...data.train.map(d=>d.x[j]))||values[k]>Math.max(...data.train.map(d=>d.x[j])));
  const reg=data.spec.type==='regression';
  const answer=reg?'£'+value.toFixed(1)+'k':(value*100).toFixed(1)+'%';
  const meaning=reg?'Estimated weekly sales.':`${data.spec.target}: ${value>=.5?'Yes':'No'} at the 50% decision threshold. This is a model estimate, not a certainty.`;
  result.innerHTML=`<div class="eyebrow">${data.spec.output} · EPOCH ${epoch}</div><strong>${answer}</strong><p>${meaning}</p><p class="help">${data.selected.map((j,k)=>data.spec.names[j]+': '+values[k]).join(' · ')}</p>`;
  if(outside.length){const note=document.createElement('p');note.className='range-warning';note.textContent='Outside the training range: '+outside.map(j=>data.spec.names[j]).join(', ')+'. This is extrapolation; predictions may be unreliable.';result.append(note)}
  if(reg&&value<0){const note=document.createElement('p');note.className='range-warning';note.textContent='This network produced negative sales, which is not a meaningful business estimate. Try more training or a different setup.';result.append(note)}
 }catch(error){result.textContent=error.message}
}
$('inferenceForm').onsubmit=e=>{e.preventDefault();predictExample()};

$('dataset').onchange=()=>{inputs();reset()};$('layers').onchange=()=>{layers();reset()};$('activation').onchange=reset;$('rate').oninput=()=>{$('rateValue').textContent=(+$('rate').value).toFixed(3);reset()};$('reset').onclick=reset;$('step').onclick=()=>{stop();advance();if(epoch<1000)$('status').textContent='Paused'};$('train').onclick=()=>{if(timer){stop();return}if(epoch>=1000)return;$('train').textContent='Ⅱ Pause training';$('status').textContent='Training';timer=setInterval(()=>advance(2),45)};$('guideToggle').onclick=()=>{$('guide').hidden=!$('guide').hidden;$('guideToggle').setAttribute('aria-expanded',String(!$('guide').hidden))};inputs();layers();reset();
if(document.modelContext?.registerTool){const lifecycle=new AbortController();try{Promise.resolve(document.modelContext.registerTool({name:'step_training',title:'Advance neural network training',description:'Pause continuous training and advance the current experiment by 1–100 epochs, updating the visible results.',inputSchema:{type:'object',properties:{epochs:{type:'integer',minimum:1,maximum:100}},required:['epochs'],additionalProperties:false},annotations:{readOnlyHint:false,untrustedContentHint:false},execute(input){if(!input||!Number.isInteger(input.epochs)||input.epochs<1||input.epochs>100||Object.keys(input).some(k=>k!=='epochs'))throw Error('epochs must be an integer from 1 to 100');stop();advance(input.epochs);return {epoch,trainingLoss:net.evaluate(data.train).loss,validationLoss:net.evaluate(data.val).loss}}},{signal:lifecycle.signal})).catch(()=>{});window.addEventListener('pagehide',()=>lifecycle.abort(),{once:true})}catch{}}
