(function(root){
 function trace(net,data,row){
  const acts=net.forward(row.input),steps=[{kind:'input',layer:0,title:'Scale the inputs',values:data.selected.map((j,k)=>({name:data.spec.names[j],raw:row.x[j],mean:data.means[k],sd:data.sd[k],value:row.input[k]}))}];
  net.w.forEach((matrix,l)=>{
   const neurons=matrix.map((weights,j)=>{const terms=weights.map((weight,k)=>({activation:acts[l][k],weight,product:acts[l][k]*weight}));const sum=terms.reduce((s,t)=>s+t.product,0),bias=net.b[l][j];return {terms,sum,bias,z:sum+bias,value:acts[l+1][j]}});
   const output=l===net.w.length-1,layer=l+1,fn=output?(data.spec.type==='regression'?'Linear':'Sigmoid'):net.activation==='tanh'?'Tanh':net.activation==='relu'?'ReLU':'Sigmoid';
   steps.push({kind:'sum',layer,title:(output?'Output layer':'Hidden layer '+layer)+': combine signals',neurons});
   steps.push({kind:'activation',layer,title:(output?'Output layer':'Hidden layer '+layer)+': apply '+fn,fn,neurons});
  });
  const raw=acts.at(-1)[0],value=data.spec.type==='regression'?raw*data.ys+data.ym:raw;
  steps.push({kind:'output',layer:net.sizes.length-1,title:'Interpret the prediction',raw,value,actual:row.y});return steps;
 }
 function rows(data){return ['train','val'].flatMap(split=>data[split].map((row,index)=>({row,split,index,id:(split==='train'?'T':'V')+String(index+1).padStart(3,'0')})))}
 function csv(data){const quote=v=>'"'+String(v).replace(/"/g,'""')+'"';const header=['Example','Split',...data.spec.names,data.spec.target];return '\uFEFF'+[header,...rows(data).map(({row,split,id})=>[id,split==='train'?'Training':'Validation',...row.x,row.y])].map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n'}
 function capture(net,data,history,epoch,settings){const t=net.evaluate(data.train),v=net.evaluate(data.val);if(!epoch||![t.loss,v.loss,v.mae,v.accuracy].every(Number.isFinite))throw Error('Train a stable network for at least one epoch before capturing.');return JSON.parse(JSON.stringify({epoch,settings,trainLoss:t.loss,valLoss:v.loss,accuracy:v.accuracy,mae:v.mae*data.ys,history,parameters:net.w.reduce((s,m)=>s+m.reduce((a,r)=>a+r.length+1,0),0)}))}
 root.LabCore={trace,rows,csv,capture};if(typeof module!=='undefined')module.exports=root.LabCore;
})(typeof window!=='undefined'?window:globalThis);
