"""Serial-pinned RGB camera sources. No policy inference or robot access."""
import configparser
import sys
import threading
import time
from pathlib import Path
import cv2
import numpy as np
from contract import ROOT
from camera_crop import center_crop_bounds

def zed_stereo_maps(filename, width, height):
    # Same pinhole/stereoRectify construction as Stereolabs Open Capture's
    # examples/include/calibration.hpp, pinned commit in evidence/manifest.json.
    cfg = configparser.ConfigParser()
    if not cfg.read(filename):
        raise FileNotFoundError(filename)
    suffix = {672: 'VGA', 1280: 'HD', 1920: 'FHD', 2208: '2K'}[width]
    matrices, distortions = [], []
    for eye in ['LEFT', 'RIGHT']:
        c = cfg[f'{eye}_CAM_{suffix}']
        vals = {k: c.getfloat(k) for k in ['fx','fy','cx','cy','k1','k2','p1','p2','k3']}
        if vals['fx'] <= 0 or vals['fy'] <= 0:
            raise ValueError('Invalid factory calibration')
        matrices.append(np.array([[vals['fx'],0,vals['cx']], [0,vals['fy'],vals['cy']], [0,0,1]],np.float64))
        # New factory files use a rational eight-coefficient lens model.
        # Dropping k4..k6 over-crops the wide-angle image severely.
        if f'{eye}_DISTO' in cfg:
            d=cfg[f'{eye}_DISTO']
            distortions.append(np.array([d.getfloat(k) for k in ['k1','k2','p1','p2','k3','k4','k5','k6']],np.float64))
        else:
            distortions.append(np.array([vals[k] for k in ['k1','k2','p1','p2','k3']],np.float64))
    s=cfg['STEREO']
    rotation=cv2.Rodrigues(np.array([s.getfloat(k+'_'+suffix.lower(),fallback=0) for k in ['rx','cv','rz']],np.float64))[0]
    translation=np.array([s.getfloat('baseline'),s.getfloat('ty_'+suffix.lower(),fallback=s.getfloat('ty',fallback=0)),s.getfloat('tz_'+suffix.lower(),fallback=s.getfloat('tz',fallback=0))],np.float64).reshape(3,1)
    r1,r2,p1,p2,_,_,_=cv2.stereoRectify(matrices[0],distortions[0],matrices[1],distortions[1],(width,height),rotation,translation,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    return {eye: cv2.initUndistortRectifyMap(matrices[i],distortions[i],rotation,projection,(width,height),cv2.CV_32FC1)
            for i,(eye,rotation,projection) in enumerate((('left',r1,p1),('right',r2,p2)))}

def zed_left_maps(filename, width, height):
    return zed_stereo_maps(filename,width,height)['left']

def discover_zed():
    sys.path.insert(0, '/home/spring/zed_camera_setup')
    from viewer import discover
    devices=discover()
    # The Mini's USB3 video device omits its serial. Match the USB2 HID on
    # the same physical controller and port route, never by enumeration order.
    usb=Path('/sys/bus/usb/devices')
    for device in devices:
        if device.get('serial') or device.get('model')!='ZED-M':continue
        node=usb/device['usb_path']
        try:
            if (node/'idProduct').read_text().strip()!='f682':continue
            bus,route=node.name.split('-',1)
            controller=(usb/('usb'+bus)).resolve().parent
            matches=[]
            for sensor in usb.glob('*-'+route):
                sensor_bus=sensor.name.split('-',1)[0]
                if (usb/('usb'+sensor_bus)).resolve().parent!=controller:continue
                try:
                    if (sensor/'idVendor').read_text().strip()=='2b03' and (sensor/'idProduct').read_text().strip()=='f681':
                        matches.append((sensor/'serial').read_text().strip())
                except OSError:continue
            if len(matches)==1 and matches[0]:
                device.update(serial=matches[0],serial_source='companion_usb2_hid_same_controller_and_port')
        except (OSError,ValueError):continue
    return devices

def realsense_color_maps(intr):
    """Pinhole output rays projected through SDK 2.58.4's distortion model."""
    import pyrealsense2 as rs
    if not any(abs(k)>1e-12 for k in intr.coeffs): return None
    if intr.model not in (rs.distortion.brown_conrady,rs.distortion.modified_brown_conrady,rs.distortion.inverse_brown_conrady):
        raise ValueError(f'Unvalidated RealSense distortion model: {intr.model}')
    yy,xx=np.mgrid[:intr.height,:intr.width].astype(np.float32)
    x=(xx-intr.ppx)/intr.fx; y=(yy-intr.ppy)/intr.fy
    k1,k2,p1,p2,k3=intr.coeffs
    r2=x*x+y*y; f=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
    if intr.model in (rs.distortion.modified_brown_conrady,rs.distortion.inverse_brown_conrady):
        x=x*f;y=y*f
        xd=x+2*p1*x*y+p2*(r2+2*x*x)
        yd=y+2*p2*x*y+p1*(r2+2*y*y)
    else:
        xd=x*f+2*p1*x*y+p2*(r2+2*x*x)
        yd=y*f+2*p2*x*y+p1*(r2+2*y*y)
    return (xd*intr.fx+intr.ppx).astype(np.float32),(yd*intr.fy+intr.ppy).astype(np.float32)

class ZedSource:
    def __init__(self, cfg):
        sys.path.insert(0, '/home/spring/zed_camera_setup')
        from capture import Capture
        devices=[d for d in discover_zed() if d['serial']==cfg['serial']]
        if len(devices)!=1:
            raise RuntimeError(f"Expected one ZED serial {cfg['serial']}, found {len(devices)}")
        self.info=devices[0]
        if cfg.get('model') and self.info['model']!=cfg['model']:
            raise RuntimeError('Configured ZED model does not match detected camera')
        self.zoom=cfg.get('digital_zoom',1.0)
        self.crop_bounds=center_crop_bounds(cfg['width_per_eye'],cfg['height'],self.zoom)
        self.eye=cfg.get('eye','left')
        self.rotation=cfg['rotation_degrees']
        if self.eye not in ('left','right','stereo'):raise ValueError('Unsupported ZED eye')
        if self.rotation not in (0,180):raise ValueError('Unsupported rotation')
        self.maps=zed_stereo_maps(ROOT/cfg['calibration'],cfg['width_per_eye'],cfg['height'])
        self.cap=Capture(self.info['device'],2*cfg['width_per_eye'],cfg['height'],cfg['fps'])
        if (self.cap.width,self.cap.height)!=(2*cfg['width_per_eye'],cfg['height']):
            self.cap.close()
            raise RuntimeError('ZED geometry changed; regenerate calibration maps')
        if abs(self.cap.fps-cfg['fps'])>.01:
            self.cap.close()
            raise RuntimeError('ZED frame rate differs from configured rate')
        self.info.update(width_per_eye=self.cap.width//2,height=self.cap.height,negotiated_fps=self.cap.fps,rectification='factory Open Capture',color='RGB',eye=self.eye,calibration=cfg['calibration'],rotation_degrees=self.rotation)
        left,top,right,bottom=self.crop_bounds
        self.info.update(digital_zoom=self.zoom,crop_xywh=[left,top,right-left,bottom-top],
                         output_width=right-left,output_height=bottom-top,
                         crop_stage='after_rectification_and_rotation_before_model_resize')
    def read(self):
        result,raw,meta=self.cap.read()
        if result != 1:
            return None
        # Reject stale queued frames as well as malformed frames rejected by C.
        stamp=meta.timestamp_us/1e6
        if not 0 <= time.monotonic()-stamp <= .25:
            return None
        stereo=cv2.cvtColor(raw,cv2.COLOR_YUV2RGB_YUYV)
        midpoint=stereo.shape[1]//2
        eyes={'left':stereo[:,:midpoint],'right':stereo[:,midpoint:]}
        selected=('left','right') if self.eye=='stereo' else (self.eye,)
        images={eye:cv2.remap(eyes[eye],*self.maps[eye],cv2.INTER_LINEAR) for eye in selected}
        if self.rotation==180:
            images={eye:cv2.rotate(image,cv2.ROTATE_180) for eye,image in images.items()}
        if self.zoom!=1:
            left,top,right,bottom=self.crop_bounds
            images={eye:image[top:bottom,left:right].copy() for eye,image in images.items()}
        return (images if self.eye=='stereo' else images[self.eye]),stamp,int(meta.sequence)
    def close(self): self.cap.close()

class RealSenseSource:
    def __init__(self,cfg):
        import pyrealsense2 as rs
        self.zoom=cfg.get('digital_zoom',1.0)
        center_crop_bounds(cfg['width'],cfg['height'],self.zoom)
        self.rs=rs
        self.pipe=rs.pipeline()
        config=rs.config()
        config.enable_device(cfg['serial'])
        config.enable_stream(rs.stream.color,cfg['width'],cfg['height'],rs.format.rgb8,cfg['fps'])
        self.profile=self.pipe.start(config)
        try:
            device=self.profile.get_device()
            name=device.get_info(rs.camera_info.name)
            if name!=cfg['model']: raise ValueError(f'Wrong wrist model: {name}')
            intr=self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            self.info={'serial':cfg['serial'],'model':name,'width':intr.width,'height':intr.height,'fps':cfg['fps'],'color':'RGB','stream':'color','intrinsics':{'fx':intr.fx,'fy':intr.fy,'ppx':intr.ppx,'ppy':intr.ppy,'coeffs':list(intr.coeffs),'model':str(intr.model)}}
            self.crop_bounds=center_crop_bounds(intr.width,intr.height,self.zoom)
            left,top,right,bottom=self.crop_bounds
            self.info.update(digital_zoom=self.zoom,crop_xywh=[left,top,right-left,bottom-top],
                             output_width=right-left,output_height=bottom-top,
                             crop_stage='after_rectification_and_rotation_before_model_resize')
            self.maps=realsense_color_maps(intr)
            self.rotation=cfg['rotation_degrees']
        except BaseException:
            self.pipe.stop()
            raise
    def read(self):
        try: f=self.pipe.wait_for_frames(1000).get_color_frame()
        except RuntimeError: return None
        if not f: return None
        received=time.monotonic()
        image=np.asanyarray(f.get_data()).copy()
        if self.maps is not None: image=cv2.remap(image,*self.maps,cv2.INTER_LINEAR)
        if self.rotation==180: image=cv2.rotate(image,cv2.ROTATE_180)
        elif self.rotation!=0: raise ValueError('Unsupported rotation')
        if self.zoom!=1:
            left,top,right,bottom=self.crop_bounds
            image=image[top:bottom,left:right].copy()
        # Reception timestamps are used for cross-device pairing. Hardware
        # timestamp domains differ; this is not hardware synchronization.
        return image,received,int(f.get_frame_number())
    def close(self): self.pipe.stop()

def wrist_source(cfg):
    if cfg['backend']=='realsense':return RealSenseSource(cfg)
    if cfg['backend']=='zed_uvc':
        if cfg.get('eye')!='left':raise ValueError('Policy wrist input must use the left eye')
        return ZedSource(cfg)
    raise ValueError('Unknown wrist camera backend')

class LatestCamera:
    def __init__(self,source):
        self.source=source;self.latest=None;self.error=None;self.count=0
        self.lock=threading.Lock();self.stop=threading.Event()
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()
    def _run(self):
        try:
            while not self.stop.is_set():
                frame=self.source.read()
                if frame is not None:
                    with self.lock: self.latest=frame;self.count+=1
        except Exception as e: self.error=e
    def get(self,max_age=.25):
        if self.error: raise self.error
        with self.lock: result=self.latest
        if result is None or time.monotonic()-result[1]>max_age:
            raise RuntimeError('No fresh camera frame')
        return result
    def close(self):
        self.stop.set();self.thread.join(timeout=3)
        if self.thread.is_alive(): raise RuntimeError('Camera worker did not stop')
        self.source.close()
