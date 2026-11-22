class Input extends AudioWorkletProcessor {process(inputs){if(inputs[0]?.[0]?.length)this.port.postMessage(inputs[0][0].slice());return true}}
registerProcessor('pcm-input',Input);
