"""Shared center-crop geometry for wrist observations and idle previews."""
import math

def center_crop_bounds(width,height,zoom=1.0):
    if type(zoom) not in (int,float) or not math.isfinite(zoom) or not 1<=zoom<=4:
        raise ValueError('Wrist digital zoom must be between 1 and 4')
    if width<1 or height<1:raise ValueError('Invalid camera dimensions')
    crop_width=max(1,round(width/zoom));crop_height=max(1,round(height/zoom))
    left=(width-crop_width)//2;top=(height-crop_height)//2
    return left,top,left+crop_width,top+crop_height
