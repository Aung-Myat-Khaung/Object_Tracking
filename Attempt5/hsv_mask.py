import cv2
import numpy as np
from config import COLORS, MIN_AREA

def find_blob(frame,range):
    
    detections = []
    final_mask = None
    for (lower,upper) in range:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(frame, lower_np, upper_np)
        if (final_mask is None):
            final_mask = mask
        else:
            final_mask = cv2.bitwise_or(final_mask,mask)
    
    #Smoothing out the image
    final_mask = cv2.erode(final_mask,None, iterations=2)
    final_mask = cv2.dilate(final_mask,None, iterations=2)
    
    return final_mask