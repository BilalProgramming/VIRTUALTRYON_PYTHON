import cv2
import numpy as np

def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    x, y = pos
    x, y = int(x), int(y) # ensure integers

    # Clipping to prevent out of bounds
    if x < 0:
        # Front image starts off-screen left
        # We need to crop the left part of imgFront
        # new msgFront x start = -x
        if -x >= wf: return imgBack # completely off
        imgFront = imgFront[:, -x:]
        x = 0
    if y < 0:
        if -y >= hf: return imgBack
        imgFront = imgFront[-y:, :]
        y = 0
        
    hf, wf, cf = imgFront.shape
    
    if x + wf > wb:
        # Crop right
        w_crop = wb - x
        imgFront = imgFront[:, :w_crop]
    if y + hf > hb:
        # Crop bottom
        h_crop = hb - y
        imgFront = imgFront[:h_crop, :]
        
    hf, wf, cf = imgFront.shape
    if hf <= 0 or wf <= 0:
        return imgBack

    # Extract Alpha
    if cf < 4:
        # No alpha channel, just overwrite
        imgBack[y:y+hf, x:x+wf] = imgFront
        return imgBack
        
    alpha_s = imgFront[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    # BGR channels
    for c in range(0, 3):
        imgBack[y:y+hf, x:x+wf, c] = (alpha_s * imgFront[:, :, c] +
                                      alpha_l * imgBack[y:y+hf, x:x+wf, c])
                                      
    return imgBack
