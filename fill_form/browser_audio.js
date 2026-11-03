() => {
  let ready = false;
  let stream=null, context=null, source=null, processor=null, mic=false, chunks=[], samples=0, micError=null, sequence=0, recordingId=0, captured=0;
  async function toggleMic() {
    if (mic) {
      mic=false;processor?.disconnect();source?.disconnect();
      stream?.getTracks().forEach(t=>t.stop());await context?.close();
      return false;
    }
    try {
      stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}});
      context=new AudioContext();await context.resume();
      // Own capture graph, no Gradio private DOM or cumulative recording blobs.
      const code=`class Capture extends AudioWorkletProcessor {process(inputs) {if(inputs[0]?.[0]) {const channels=inputs[0]; const mono=new Float32Array(channels[0].length);for(const c of channels)for(let i=0;i<c.length;i++)mono[i]+=c[i]/channels.length;this.port.postMessage(mono);}return true;}}registerProcessor('capture',Capture);`;
      const url=URL.createObjectURL(new Blob([code],{type:'application/javascript'}));
      try {await context.audioWorklet.addModule(url);} finally {URL.revokeObjectURL(url);}
      source=context.createMediaStreamSource(stream);
      processor=new AudioWorkletNode(context,'capture');
      processor.port.onmessage=({data})=>{
        if(!mic)return;
        if(samples+data.length>context.sampleRate*3){micError='Audio buffer đầy; dừng mic để tránh mất dữ liệu';toggleMic();return;}
        chunks.push(data);samples+=data.length;captured+=data.length;
      };
      source.connect(processor);processor.connect(context.destination);mic=true;recordingId++;micError=null;return true;
    } catch(e) {
      micError=e.name+': '+e.message;stream?.getTracks().forEach(t=>t.stop());await context?.close();mic=false;return false;
    }
  }
  function drain(force=false) {
    if (!ready && !force) return '';
    ready = true;
    let data='';const rate=context?.sampleRate||16000;
    if(samples){const bytes=new Uint8Array(samples*2);const view=new DataView(bytes.buffer);let offset=0;
      for(const c of chunks)for(const x of c){view.setInt16(offset,Math.round(Math.max(-1,Math.min(1,x))*32767),true);offset+=2;}
      for(let i=0;i<bytes.length;i+=8192)data+=String.fromCharCode(...bytes.subarray(i,i+8192));
    }
    chunks=[];samples=0;const err=micError;micError=null;
    return JSON.stringify({sequence:++sequence,recording_id:recordingId,rate,audio:btoa(data),mic,error:err});
  }
  window.fillForm = {
    toggleMic, drain,
    stats:()=>({mic,samples,captured,state:context?.state,rate:context?.sampleRate}),
    apply(snapshot) {
      if(!snapshot)return;
      ready=true;
      if(snapshot.capture_error && mic){micError=snapshot.capture_error;toggleMic();}
    }
  };
}
