import cv2
import time
import queue
import threading
import numpy as np

from frame import Frame_Grabber
from config import FRAME_SIZE,COLORS, MAX_MISSES, MIN_AREA, MAX_DISTANCE, WIDTH,HEIGHT
from hsv_mask import find_blob
from canny_mask import detect_circle
from hungarian import assign
from tracker import Kalman_tracker


###Custom function for thread
def _check_mask_area(x, y, r, contour):

    circle_mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.circle(circle_mask, (x, y), r, 255, -1)

    contour_mask = np.zeros_like(circle_mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    overlap   = cv2.bitwise_and(circle_mask, contour_mask)
    cov_ratio = cv2.countNonZero(overlap) / cv2.countNonZero(circle_mask)
    return cov_ratio

def canny_hough(frame,circle_queue):
    circle_queue.put(detect_circle(frame))

def hsv_mask(frame,hsv_queue):
    hsv_queue.put(find_blob(frame, [(COLORS['green']['lower'],COLORS['green']['higher'])]))

def check_center(hsv_queue,circle_queue):
    hsv_frame = hsv_queue.get()
    circles = circle_queue.get()

    contour, _ = cv2.findContours(hsv_frame,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    used_circ = set()
    bbox = []
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area >2100:
            double = False
            found = []
            for i,(x,y,r) in enumerate(circles):
                if i in used_circ:
                    continue
                elif cv2.pointPolygonTest(cnt, (x, y), measureDist=False) >= 1:
                    used_circ.add(i)
                    if _check_mask_area(x,y,r,cnt)>0.75:
                        found.append((x-r,y-r,2*r,2*r))
            bbox.extend(found)
        else:
            if area>MIN_AREA:
                x,y,w,h = cv2.boundingRect(cnt)
                bbox.append((x,y,w,h))
    return bbox

    



trackers = []
tracker_id = 1
stream = Frame_Grabber(1)
stream.start()

hsv_queue = queue.Queue(maxsize=1)
circle_queue = queue.Queue(maxsize=1)
while True:
    frame = stream.read_data()  
    if frame is None or frame.size == 0:
        continue 
    frame = cv2.resize(frame,FRAME_SIZE)
    t1 = threading.Thread(target=hsv_mask,args=((frame,hsv_queue)))
    t2 = threading.Thread(target=canny_hough,args=((frame,circle_queue)))

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    detections = check_center(hsv_queue,circle_queue)
    p_center = [t.predict_center() for t in trackers]

    mt, un_t, un_d = assign(trackers,detections)
    for r, c in mt:
        trackers[r].update_loss(frame,detections[c])
        trackers[r].misses = 0

    for r in un_t:
        trackers[r].misses += 1

    for c in un_d:
        trackers.append(Kalman_tracker(tracker_id, detections[c],frame))
        tracker_id += 1

    trackers = [t for t in trackers if t.misses<MAX_MISSES]
    for t in trackers:
        (cx, cy), ok = t.update(frame)
        color = (0,255,0) if ok else (0,0,255)
        x,y,w,h = t.bbox
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"ID {t.id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imshow("Multi‑Ball Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
stream.stop()
cv2.destroyAllWindows()








