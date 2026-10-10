/* Present the existing experiment in focused views; moving nodes preserves state. */
(function(){
 const q=s=>document.querySelector(s), main=q('main'), old=q('#workspace'), results=q('.results'), config=q('.config'), networkPanel=q('.network-panel'), inspector=$('networkInspector');
 const nav=q('.lab-nav');nav.setAttribute('role','tablist');nav.setAttribute('aria-label','Experiment tasks');
 const tasks=[['train','Train'],['trace','Trace'],['predict','Predict'],['compare','Compare'],['data','Data'],['activities','Activities']];
 nav.innerHTML=tasks.map(([id,name])=>`<button id="tab-${id}" role="tab" aria-controls="view-${id}" aria-selected="false" tabindex="-1">${name}</button>`).join('');
 const summary=document.createElement('div');summary.className='experiment-summary';summary.innerHTML='<button id="setupToggle" class="secondary" aria-expanded="false" aria-controls="experimentSetup">Experiment setup</button><p id="experimentSummary"></p>';
 nav.before(summary);config.id='experimentSetup';const views=document.createElement('div');views.className='task-views';old.append(views);
 const panels={};for(const [id] of tasks){const p=document.createElement('section');p.id='view-'+id;p.className='task-view';p.setAttribute('role','tabpanel');p.setAttribute('aria-labelledby','tab-'+id);p.tabIndex=0;p.hidden=true;panels[id]=p;views.append(p)}
 const trainStudy=document.createElement('div');trainStudy.className='study-grid';const trainDiagram=document.createElement('div');trainDiagram.className='diagram-slot';trainStudy.append(trainDiagram,inspector);panels.train.append(trainStudy,q('.metrics'),q('.chart-grid'),q('.insight'));
 const traceStudy=document.createElement('div');traceStudy.className='study-grid trace-study';const traceDiagram=document.createElement('div');traceDiagram.className='diagram-slot';traceStudy.append(traceDiagram,$('tracePanel'));panels.trace.append(traceStudy);
 panels.predict.append(q('.inference-panel'));panels.compare.append($('comparePanel'));panels.data.append($('datasetPanel'));panels.activities.append($('activityPanel'));
 const evidence=document.createElement('div');evidence.id='activityComparison';panels.activities.append(evidence);
 const bar=q('.train-bar');bar.classList.add('persistent-training');const metrics=document.createElement('span');metrics.id='toolbarMetrics';metrics.className='toolbar-metrics';bar.append(metrics);main.append(bar);
 const activations=document.createElement('p');activations.id='layerActivations';activations.className='activation-summary';q('.network-controls').after(activations);
 const all=document.createElement('label');all.className='values-toggle';all.innerHTML='<input id="allWeights" type="checkbox"> All weights';q('.network-controls').append(all);$('allWeights').onchange=network;
 $('valuesHint').textContent='Select a neuron to inspect its incoming weights and calculation. Enable All weights to label every connection. Selecting pauses training.';
 $('tracePanel').querySelector(':scope > p').textContent='Follow one example through the network beside this explanation. Weights stay fixed during a prediction trace.';
 const setupDialog=document.createElement('dialog');setupDialog.className='setup-dialog';setupDialog.setAttribute('aria-label','Experiment setup');setupDialog.innerHTML='<button class="secondary" id="closeSetup">Done</button>';document.body.append(setupDialog);
 let active='train',expanded=false;
 function closeSetup(){if(setupDialog.open)setupDialog.close();old.prepend(config);config.hidden=!expanded;$('setupToggle').setAttribute('aria-expanded',String(expanded));}
 setupDialog.addEventListener('close',()=>{old.prepend(config);config.hidden=!expanded;$('setupToggle').setAttribute('aria-expanded',String(expanded));$('setupToggle').focus()});
 $('closeSetup').onclick=closeSetup;
 $('setupToggle').onclick=()=>{if(matchMedia('(max-width:900px)').matches){config.hidden=false;setupDialog.append(config);setupDialog.showModal();$('setupToggle').setAttribute('aria-expanded','true')}else{expanded=!expanded;config.hidden=!expanded;old.classList.toggle('setup-expanded',expanded);$('setupToggle').setAttribute('aria-expanded',String(expanded))}};
 function update(){
  const activation={tanh:'Tanh',relu:'ReLU',sigmoid:'Sigmoid'}[net.activation],output=data.spec.type==='regression'?'Linear':'Sigmoid';
  $('experimentSummary').textContent=$('dataset').selectedOptions[0].textContent+' · '+data.selected.length+' inputs · '+(hidden.length?hidden.join(' → ')+' hidden neurons':'no hidden layers')+' · learning rate '+$('rate').value;
  activations.textContent='Hidden: '+(hidden.length?activation:'none')+' · Output: '+output;
  metrics.textContent='Loss '+$('valLoss').textContent+' · '+$('scoreLabel').textContent+' '+$('score').textContent;
  $('allWeights').disabled=!$('showValues').checked;
  trainStudy.classList.toggle('has-inspector',!inspector.hidden);
 }
 function show(id,focus=false){
  if(!panels[id])return;if(id!==active){stop();if(active==='trace')Lab.endTrace();}active=id;
  for(const [key] of tasks){panels[key].hidden=key!==id;const b=$('tab-'+key);b.setAttribute('aria-selected',String(key===id));b.tabIndex=key===id?0:-1;}
  (id==='trace'?traceDiagram:trainDiagram).append(networkPanel);
  if(id==='activities')evidence.append($('comparePanel'));else panels.compare.append($('comparePanel'));
  bar.hidden=!['train','trace','activities'].includes(id);
  document.body.dataset.task=id;network();update();
  if(focus)$('tab-'+id).focus();
 }
 nav.onclick=e=>{const b=e.target.closest('[role=tab]');if(b)show(b.id.slice(4))};
 nav.onkeydown=e=>{const keys=['ArrowLeft','ArrowRight','Home','End'];if(!keys.includes(e.key))return;e.preventDefault();let i=tasks.findIndex(t=>t[0]===active);i=e.key==='Home'?0:e.key==='End'?tasks.length-1:(i+(e.key==='ArrowRight'?1:-1)+tasks.length)%tasks.length;show(tasks[i][0],true)};
 // Data-row trace opens the trace view before the original handler runs.
 $('dataTable').addEventListener('click',e=>{if(e.target.closest('.trace-row'))show('trace')},true);
 const previousTrainingChanged=Lab.trainingChanged;Lab.trainingChanged=()=>{previousTrainingChanged();update()};
 const observer=new MutationObserver(update);observer.observe($('epoch'),{childList:true});observer.observe(inspector,{attributes:true,attributeFilter:['hidden']});
 $('showValues').addEventListener('change',update);$('inputs').addEventListener('change',update);$('dataset').addEventListener('change',update);$('activation').addEventListener('change',update);$('layers').addEventListener('change',update);$('rate').addEventListener('input',update);
 window.Workspace={show,update};config.hidden=true;show('train');results.remove();
})();
