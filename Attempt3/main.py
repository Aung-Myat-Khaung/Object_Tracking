import cv2
import numpy as np
from detector import detect_cnt
from tracker import Kalman_tracker
from config import FRAME_SIZE, MAX_MISS, MAX_DISTANCE, COLORS
from frame import Frame_Grabber

stream = Frame_Grabber(1)
stream.start()
trackers = []
id = 0

while True:
    raw = stream.read_data()    
    if raw is None or raw.size == 0:
        continue  
    frame = cv2.resize(raw,FRAME_SIZE)
    detections = detect_cnt(frame, [(COLORS['green']['lower'],COLORS['green']['higher'])])

    p_center = [t.predict_center() for t in trackers]

    used_det = set()
    for t_idx, tracker in enumerate(trackers):
        best_i = None; best_d = MAX_DISTANCE
        for d_idx, (bbox) in enumerate(detections):
            if d_idx in used_det: continue
            cx_d, cy_d = bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2
            px, py = p_center[t_idx]
            d = np.hypot(px-cx_d, py-cy_d)
            if d < best_d:
                best_d, best_i = d, d_idx
        if best_i is not None:
            tracker.update_post(frame, detections[best_i])
            used_det.add(best_i)
        else:
            tracker.update_post(frame)

    for i, (bbox) in enumerate(detections):
        if i in used_det or len(trackers) >= 5:
           continue
        trackers.append(Kalman_tracker(id, bbox, frame))
        id += 1
    
    trackers = [t for t in trackers if t.misses < MAX_MISS]

    for t in trackers:
        (cx, cy), ok = t.update_post(frame, detections[best_i])  # ensure each tracker advances exactly once per frame
        x,y,w,h = t.bbox
        colour = (0,255,0) if ok else (0,0,255)
        cv2.rectangle(frame, (x,y), (x+w,y+h), colour, 2)
        cv2.putText(frame, f"ID {t.id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.imshow("Multi‑Ball Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
stream.stop()
cv2.destroyAllWindows()