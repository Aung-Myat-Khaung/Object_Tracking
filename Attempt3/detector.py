import cv2
import numpy as np
from config import COLORS, MIN_AREA

def detect_cnt(frame,range):
    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    detections = []
    final_mask = None
    for (lower,upper) in range:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower_np, upper_np)
        if (final_mask is None):
            final_mask = mask
        else:
            final_mask = cv2.bitwise_or(final_mask,mask)
    
    #Smoothing out the image
    final_mask = cv2.erode(final_mask,None, iterations=2)
    final_mask = cv2.dilate(final_mask,None, iterations=2)

    countors, _  = cv2.findContours(final_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in countors:
        area = cv2.contourArea(cnt)
        if area>MIN_AREA:
            x,y,w,h = cv2.boundingRect(cnt)
            detections.append((x,y,w,h))
    return detections