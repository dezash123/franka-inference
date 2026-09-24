"""Bounded-wait V4L2 reader with driver error and payload-size validation."""
import ctypes as C
import os
from pathlib import Path
import numpy as np

class Metadata(C.Structure):
    _fields_=[('sequence',C.c_uint32),('flags',C.c_uint32),
              ('bytesused',C.c_uint32),('timestamp_us',C.c_uint64)]

LIB=C.CDLL(str(Path(__file__).resolve().parent/'build/libcapture_v4l2.so'),use_errno=True)
LIB.capture_has_yuyv.argtypes=[C.c_char_p]
LIB.capture_has_yuyv.restype=C.c_int
LIB.capture_open.argtypes=[C.c_char_p,C.c_uint32,C.c_uint32,C.c_uint32,C.POINTER(C.c_uint32)]
LIB.capture_open.restype=C.c_void_p
LIB.capture_read.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(Metadata)]
LIB.capture_read.restype=C.c_int
LIB.capture_close.argtypes=[C.c_void_p]
LIB.capture_close.restype=None

def has_yuyv(device):
    return bool(LIB.capture_has_yuyv(os.fsencode(device)))

class Capture:
    def __init__(self,device,width,height,fps):
        geometry=(C.c_uint32*4)()
        self.handle=LIB.capture_open(os.fsencode(device),width,height,fps,geometry)
        if not self.handle:
            raise OSError(C.get_errno(),os.strerror(C.get_errno()),device)
        self.width,self.height,n,d=geometry
        self.fps=d/n if n else 0
        self.raw=np.empty((self.height,self.width,2),dtype=np.uint8)
        self.meta=Metadata()

    def read(self):
        result=LIB.capture_read(self.handle,self.raw.ctypes.data,self.raw.nbytes,C.byref(self.meta))
        if result<0:
            raise OSError(C.get_errno(),os.strerror(C.get_errno()))
        return result,self.raw,self.meta

    def close(self):
        if self.handle:
            LIB.capture_close(self.handle)
            self.handle=None
