import cv2
import queue
import threading
import numpy as np

from frame import Frame_Grabber
from config import FRAME_SIZE, COLORS, MAX_MISSES, MIN_AREA, WIDTH, HEIGHT, MAX_TRACKERS
from hsv_mask import find_blob
from canny_mask import detect_circle
from hungarian import assign
from tracker import Kalman_tracker





def _check_mask_area(x, y, r, contour):
    """Return the ratio of a circle that is filled by *contour* (coverage test)."""
    circle_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    cv2.circle(circle_mask, (x, y), r, 255, -1)

    contour_mask = np.zeros_like(circle_mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    overlap = cv2.bitwise_and(circle_mask, contour_mask)
    return cv2.countNonZero(overlap) / cv2.countNonZero(circle_mask)


def canny_hough(frame, circle_queue):

    circle_queue.put(detect_circle(frame))


def hsv_mask(frame, hsv_queue):

    hsv_queue.put(find_blob(frame, [(COLORS['green']['lower'], COLORS['green']['higher'])]))


def fuse_detections(hsv_queue, circle_queue):

    hsv_frame = hsv_queue.get()
    circles = circle_queue.get()

    contours, _ = cv2.findContours(hsv_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    used_circ = set()
    bboxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2100:
            for i, (x, y, r) in enumerate(circles):
                if i in used_circ:
                    continue
                if cv2.pointPolygonTest(cnt, (x, y), False) >= 0 and _check_mask_area(x, y, r, cnt) > 0.75:
                    used_circ.add(i)
                    bboxes.append((x - r, y - r, 2 * r, 2 * r))
        elif area > MIN_AREA:
            bboxes.append(cv2.boundingRect(cnt))

    return bboxes


stream = Frame_Grabber(1)
stream.start()

trackers = []                         
id_pool = set(range(1, MAX_TRACKERS + 1))  

hsv_queue = queue.Queue(maxsize=1)
circle_queue = queue.Queue(maxsize=1)

while True:
    frame = stream.read_data()
    if frame is None or frame.size == 0:
        continue
    frame = cv2.resize(frame, FRAME_SIZE)
    t1 = threading.Thread(target=hsv_mask, args=(frame, hsv_queue))
    t2 = threading.Thread(target=canny_hough, args=(frame, circle_queue))
    t1.start(); t2.start(); t1.join(); t2.join()
    detections = fuse_detections(hsv_queue, circle_queue)
    mt, un_t, un_d = assign(trackers, detections)

    for r, c in mt:
        trackers[r].update_loss(frame, detections[c])
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
        trackers.append(Kalman_tracker(new_id, detections[c], frame))
        space_left -= 1

    alive_trackers = []
    for t in trackers:
        if t.misses < MAX_MISSES:
            alive_trackers.append(t)
        else:
            id_pool.add(t.id)  
    trackers = alive_trackers

    for t in trackers:
        (cx, cy), ok = t.update(frame)
        x, y, w, h = t.bbox
        colour = (0, 255, 0) if ok else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(frame, f"ID {t.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    cv2.imshow("Multi-Ball Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()