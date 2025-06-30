import cv2
import time
import queue
import threading
import numpy

from frame import Frame_Grabber
from config import FRAME_SIZE,COLORS, MAX_MISSES
from hsv_mask import find_blob
from canny_mask import detect_circle
from hungarian import assign
from tracker import Kalman_tracker

stream = Frame_Grabber(1)
stream.start()
trackers = []
tracker_id = 0

while True:
    frame = stream.read_data()  
    if frame is None or frame.size == 0:
        continue 
    frame = cv2.resize(frame,FRAME_SIZE)
    final_mask = find_blob(frame, [(COLORS['green']['lower'],COLORS['green']['higher'])])
    cv2.imshow('hsv mask',final_mask)
    detection = detect_circle(frame)

    mt, un_t, un_d = assign(trackers,detection)
    for r, c in mt:
        trackers[r].update(frame)
        trackers[r].misses = 0
    for r in un_t:
        trackers[r].misses += 1
    for c in un_d:
        trackers.append(Kalman_tracker(tracker_id, detection[c],frame))
        tracker_id += 1
    trackers = [t for t in trackers if t.misses<MAX_MISSES]
    for t in trackers:
        (cx, cy), ok = t.update(frame)
        color = (0,255,0) if ok else (0,0,255)
        if ok:
            cv2.circle(frame,(cx,cy),t.r,color)
            cv2.putText(frame, f"ID {t.id}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.circle(frame,(cx,cy),8,color)
    cv2.imshow("Multi_Tracker",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

stream.stop()
cv2.destroyAllWindows()