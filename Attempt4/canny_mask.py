import cv2
import numpy as np
from config import CANNY_HIGH,CANNY_LOW
def detect_circle(frame):
    circle_detection = []
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frame = cv2.Canny(frame,CANNY_LOW,CANNY_HIGH)
    circle = cv2.HoughCircles(
        frame, cv2.HOUGH_GRADIENT, dp=1.0, minDist=20,
        param1=70, param2=30, minRadius=20, maxRadius=50)
    
    if circle is not None:
        circles = np.uint16(np.around(circle[0]))
        for (x, y, r) in circles:
            circle_detection.append((int(x),int(y),int(r)))
    return circle_detection