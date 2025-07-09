import cv2
import queue
import threading
import numpy as np

from frame import Frame_Grabber
from config import FRAME_SIZE, COLORS, MAX_MISSES, MIN_AREA, MAX_TRACKERS, NUM_COLOR
from hsv_mask import find_blob
from canny_mask import detect_circle
from hungarian import assign
from tracker import Kalman_tracker

def _check_mask_area(x, y, r, contour,h,w):
    
    if r <= 0 or not (0 <= x < w and 0 <= y < h):
        return 0.0       
    
    x, y, r = int(x), int(y), int(r)
    circle_mask = np.zeros((h,w), dtype=np.uint8)
    cv2.circle(circle_mask, (x, y), r, 255, -1)

    contour_mask = np.zeros_like(circle_mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    overlap = cv2.bitwise_and(circle_mask, contour_mask)
    return cv2.countNonZero(overlap) / cv2.countNonZero(circle_mask)

def canny_hough(frame, circle_queue):

    circle_queue.put(detect_circle(frame))


def hsv_mask(frame, hsv_queue,color_name):

    hsv_queue.put((find_blob(frame, [(COLORS[color_name]['lower'], COLORS[color_name]['higher'])]),color_name))


def fuse_detections(hsv_queue, circle_queue,h,w):

    hsv_frame = [hsv_queue.get() for _ in range(NUM_COLOR)]
    circles = circle_queue.get()

    used_circle = set()
    bboxes = []
    for mask, color in hsv_frame:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            matched = False
            if area > MIN_AREA:
                for i, (x, y, r) in enumerate(circles):
                    if i in used_circle:
                        continue
                    if cv2.pointPolygonTest(cnt, (x, y), False) >= 0 and _check_mask_area(x, y, r, cnt, h, w) > 0.75:
                        used_circle.add(i)
                        matched = True
                        bboxes.append(((x - r, y - r, 2 * r, 2 * r), color))
                        break
            if not matched and area>MIN_AREA:
                x_post, y_post, w, h = cv2.boundingRect(cnt)
                bboxes.append(((x_post, y_post, w, h), color))

    return bboxes


stream = Frame_Grabber(1)
stream.start()
frame_tracker = 0
trackers = []                         
id_pool = set(range(1, MAX_TRACKERS + 1))  

hsv_queue = queue.Queue(maxsize=4)
circle_queue = queue.Queue(maxsize=1)

while True:
    frame = stream.read_data()
    if frame is None or frame.size == 0:
        continue
    frame = cv2.resize(frame, FRAME_SIZE)
    h,w = frame.shape[:2]
    t2 = threading.Thread(target=canny_hough, args=(frame, circle_queue))
    t2.start()
    threads = []
    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    for color_name in COLORS:
        t = threading.Thread(target=hsv_mask,args=(hsv_img,hsv_queue,color_name,))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    t2.join()

    detections = fuse_detections(hsv_queue, circle_queue,h,w)
    mt, un_t, un_d = assign(trackers, [d[0] for d in detections])

    for r, c in mt:
        trackers[r].update_loss(frame, detections[c][0],detections[c][1])
        trackers[r].misses = 0

    for r in un_t:
        trackers[r].misses += 1

    num_live = len(trackers) - len(un_t)         
    space_left = min(len(detections) - num_live,  
                     MAX_TRACKERS - len(trackers))  
    for c in un_d:
        if space_left <= 0 or not id_pool:
            break  
        new_id = id_pool.pop()
        bbox, color = detections[c]  
        trackers.append(Kalman_tracker(new_id, detections[c][0], frame,detections[c][1]))
        space_left -= 1

    alive_trackers = []
    for t in trackers:
        if t.misses < MAX_MISSES:
            alive_trackers.append(t)
        else:
            id_pool.add(t.id)  
    trackers = alive_trackers

    for t in trackers:
        if frame_tracker:
            (cx, cy), ok = t.update(frame)
        else:
            cx,cy = t.predict_center()
            ok = False
        x, y, w, h = t.bbox
        colour = (0, 255, 0) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(frame, f"Color: {t.color}, ID {t.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.imshow("Multi-Ball Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()