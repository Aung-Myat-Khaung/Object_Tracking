import cv2
import numpy as np
from detector import detect_cnt
from tracker import Kalman_tracker
from config import FRAME_SIZE, MAX_MISS, MAX_DISTANCE, COLORS
from frame import Frame_Grabber

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
    
    used_detection = set()
    for t_idx, tracker in enumerate(trackers):
        diff_d = MAX_DISTANCE
        state = None
        found = False
        for d_idx, detection in enumerate(detections):
            if d_idx in used_detection or found == True:
                continue
            else:
                cx, cy = detections[d_idx][0]+detections[d_idx][2]/2, detections[d_idx][1]+detections[d_idx][3]/2
                px, py = p_center[t_idx]
                dist = np.hypot(cx-px,cy-py)  
                if (dist<MAX_DISTANCE):
                    state = d_idx
                    diff_d = dist
                    found = True
            if state is not None:
                tracker.update_detection(frame,detections[d_idx])
                used_detection.add(d_idx)

            else:
                tracker.update_post(frame)

    for i, (bbox) in enumerate(detections):
        if i not in used_detection:
            trackers.append(Kalman_tracker(tracker_id, bbox,frame))
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