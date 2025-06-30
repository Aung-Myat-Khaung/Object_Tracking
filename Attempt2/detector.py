import cv2
import numpy as np
class ColorDetection():
    def __init__(self, range,color):
        """
        Constructor

        Initialize the color range and color type
        """
        self.color_range = range
        self.color = color

    def detections(self,frame):
        """
        
        """

        #Converting to HSV color type
        hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        #Finding area within the color range
        final_mask = None
        for (lower,upper) in self.color_range:
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

        #Finding contour and bounding box
        contours, _ = cv2.findContours(final_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area>300:
                x,y,w,h = cv2.boundingRect(cnt)
                return (x,y,w,h)
        return None