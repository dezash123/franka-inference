"""Entire pi0.5 neural inference in OpenVINO FP16 on the B580."""
import json,time,threading
from pathlib import Path
import numpy as np
import openvino as ov
from contract import ROOT,validate_observation,settings
from ov_common import transforms,prepared

class OpenVINOPolicy:
    def __init__(self):
        self.input_transform,self.output_transform=transforms()
        self.steps=int(settings()['policy']['denoise_steps'])
        if not 1<=self.steps<=50:raise ValueError('Denoising steps must be between 1 and 50')
        self.lock=threading.Lock();self.rng=np.random.default_rng()
        self.core=ov.Core();self.context=self.core.get_default_context('GPU')
        self.device_name=self.core.get_property('GPU','FULL_DEVICE_NAME')
        if 'B580' not in self.device_name:raise RuntimeError(f'Unexpected GPU: {self.device_name}')
        root=ROOT/'openvino_fp16';(root/'cache').mkdir(exist_ok=True)
        props={'PERFORMANCE_HINT':'LATENCY','NUM_STREAMS':ov.properties.streams.Num(1),'INFERENCE_PRECISION_HINT':ov.Type.f16,
               'DYNAMIC_QUANTIZATION_GROUP_SIZE':0,'KV_CACHE_PRECISION':'f16','CACHE_DIR':str(root/'cache')}
        self.models={};self.requests={};self.remote=[];self.runtime={}
        for stage in ('vision','prefix','denoise'):
            start=time.monotonic();print('COMPILING',stage,flush=True)
            model=self.core.read_model(root/f'{stage}.xml')
            # A finite FP16-representable mask prevents infinities on padded queries.
            # Masked keys remain exactly zero after softmax for normal attention scores.
            for op in model.get_ordered_ops():
                if op.get_type_name()=='Constant' and op.get_element_type()==ov.Type.f32 and int(np.prod(op.get_output_shape(0)))==1:
                    val=np.asarray(op.get_data())
                    if float(val.reshape(-1)[0]) < -1e30:
                        replacement=ov.opset13.constant(np.full(val.shape,-10000.,np.float32))
                        op.output(0).replace(replacement.output(0))
            model.validate_nodes_and_infer_types()
            compiled=self.core.compile_model(model,self.context,props)
            self.models[stage]=compiled;self.requests[stage]=compiled.create_infer_request()
            self.runtime[stage]={'compile_s':time.monotonic()-start,
                                 'execution_devices':compiled.get_property('EXECUTION_DEVICES'),
                                 'inference_precision':str(compiled.get_property('INFERENCE_PRECISION_HINT')),
                                 'dynamic_quantization_group_size':compiled.get_property('DYNAMIC_QUANTIZATION_GROUP_SIZE')}
            ov.serialize(compiled.get_runtime_model(),root/f'{stage}-runtime.xml')
            print('COMPILED',stage,self.runtime[stage],flush=True)
        def connect(source,out_index,destination,in_index):
            port=self.models[source].output(out_index)
            tensor=self.context.create_tensor(port.element_type,ov.Shape(port.shape),{})
            self.requests[source].set_tensor(port.any_name,tensor)
            self.requests[destination].set_tensor(self.models[destination].input(in_index).any_name,tensor)
            self.remote.append(tensor)
        connect('vision',0,'prefix',0)
        connect('prefix',0,'denoise',3);connect('prefix',1,'denoise',4)
        (root/'runtime.json').write_text(json.dumps({'openvino':ov.__version__,'device':self.device_name,'graphs':self.runtime,
                                                  'kv_and_vision_buffers':'GPU remote tensors','denoise_steps':self.steps},indent=2))
    def infer(self,obs,*,noise=None):
        with self.lock:return self._infer(obs,noise=noise)
    def _infer(self,obs,*,noise=None):
        begin=time.perf_counter();validate_observation(obs)
        data,images,tokens,pad=prepared(obs,self.input_transform)
        t0=time.perf_counter()
        vr,pr,dr=(self.requests[k] for k in ('vision','prefix','denoise'))
        vr.set_input_tensor(0,ov.Tensor(images));vr.start_async();vr.wait()
        t1=time.perf_counter()
        pr.set_input_tensor(1,ov.Tensor(tokens));pr.set_input_tensor(2,ov.Tensor(pad));pr.start_async();pr.wait()
        t2=time.perf_counter()
        dr.set_input_tensor(2,ov.Tensor(pad))
        x=self.rng.standard_normal((1,15,32),dtype=np.float32) if noise is None else np.asarray(noise,np.float32).reshape(1,15,32).copy()
        for i in range(self.steps):
            dr.set_input_tensor(0,ov.Tensor(x));dr.set_input_tensor(1,ov.Tensor(np.array([1.-i/self.steps],np.float32)))
            dr.infer();velocity=dr.get_output_tensor(0).data
            x=x-np.float32(1./self.steps)*velocity
        t3=time.perf_counter()
        result=self.output_transform({'state':data['state'],'actions':x[0]})
        actions=np.asarray(result['actions'])
        if actions.shape!=(15,8) or not np.isfinite(actions).all():raise RuntimeError('OpenVINO produced invalid actions')
        done=time.perf_counter()
        result['policy_timing']={'infer_ms':(t3-t0)*1000,'preprocess_ms':(t0-begin)*1000,'vision_ms':(t1-t0)*1000,
                                 'prefix_ms':(t2-t1)*1000,'denoise_ms':(t3-t2)*1000,'total_ms':(done-begin)*1000}
        return result
