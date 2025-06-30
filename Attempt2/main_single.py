import cv2
from detector import ColorDetection
from tracker import Ball_Tracker
from create_tracker import CreateTracker
import numpy as np

from config import MAX_MISS, TRACKER_TYPE, COLORS, MAX_DIST, FRAME_SIZE

detectors = ColorDetection([(COLORS['green']['lower-hsv'],COLORS['green']['higher-hsv'])],'green')
tracker = None
id = 0
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera not opened")
    SystemExit

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, FRAME_SIZE)
    if ret is None:
        print("Cannot fetch frame.")
        SystemExit

    boxes = detectors.detections(frame)

    if tracker is None and boxes is not None:
        tracker = Ball_Tracker(frame,boxes)
    
    if tracker is not None:
        if boxes is not None:
            dx, dy = boxes[0]+boxes[2]/2 , boxes[1]+boxes[3]/2
            px, py = tracker.predict()
            if (np.hypot(dx-px,dy-py)>MAX_DIST):
                tracker = Ball_Tracker(frame,boxes)
        ok, (cx, cy, r) = tracker.update(frame)
        colour = (0,255,0) if ok else (0,0,255)
        cv2.circle(frame, (int(dx), int(dy)), max(int(r), 8), colour, 2)
        cv2.putText(frame, f"ID {id}", (int(cx), int(cy-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        
        
        if tracker.misses > MAX_MISS:
            tracker = None 
            id += 1 

    cv2.imshow("Single Ball KCF+Kalman", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()