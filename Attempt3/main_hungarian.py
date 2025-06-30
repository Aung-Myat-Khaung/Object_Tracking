import cv2
import numpy as np
from detector import detect_cnt
from tracker import Kalman_tracker
from config import FRAME_SIZE, MAX_MISS, MAX_DISTANCE, COLORS
from frame import Frame_Grabber
from hungarian import assign

stream = Frame_Grabber(1)
stream.start()
trackers = []
tracker_id = 0

while True:
    raw = stream.read_data()  
    if raw is None or raw.size == 0:
        continue  
    frame = cv2.resize(raw,FRAME_SIZE)
    detections = detect_cnt(frame, [(COLORS['green']['lower'],COLORS['green']['higher'])])
    
    p_center = [t.predict_center() for t in trackers]

    mt, un_t, un_d = assign(trackers,detections,MAX_DISTANCE)
    for r, c in mt:
        trackers[r].update_detection(frame,detections[c])
        trackers[r].misses = 0

    for r in un_t:
        trackers[r].misses += 1

    for c in un_d:
        trackers.append(Kalman_tracker(tracker_id, detections[c],frame))
        tracker_id += 1

    trackers = [t for t in trackers if t.misses<MAX_MISS]
    for t in trackers:
        (cx, cy), ok = t.update_post(frame)
        color = (0,255,0) if ok else (0,0,255)
        x,y,w,h = t.bbox
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"ID {t.id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("Multi‑Ball Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
stream.stop()
cv2.destroyAllWindows()