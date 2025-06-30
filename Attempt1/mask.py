import numpy as np
import cv2
def get_mask(hsv_img, range):
    
    final_mask = None
    for lower, upper in range:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower_np, upper_np)
        if final_mask is None:
            final_mask = mask
        else:
            final_mask = cv2.bitwise_or(final_mask, mask)
    mask = cv2.erode(mask,None, iterations=2)
    mask = cv2.dilate(mask,None,iterations=2)
    return mask