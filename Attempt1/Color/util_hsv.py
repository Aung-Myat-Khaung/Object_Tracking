import numpy as np
import cv2
def get_HSVmask(hsv_img, hsv_ranges):
    final_mask = None
    for lower, upper in hsv_ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv_img, lower_np, upper_np)
        if final_mask is None:
            final_mask = mask
        else:
            final_mask = cv2.bitwise_or(final_mask, mask)

    kernel = np.ones((7, 7), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    return final_mask

