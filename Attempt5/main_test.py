import cv2
import queue
import threading
import numpy as np

from frame import Frame_Grabber
from config import FRAME_SIZE, COLORS, MAX_MISSES, MIN_AREA, WIDTH, HEIGHT, MAX_TRACKERS, NUM_COLOR
from hsv_mask import find_blob
from canny_mask import detect_circle
from hungarian import assign
from tracker import Kalman_tracker

import time
from collections import defaultdict


class StageTimer:
    """Simple multi-stage stopwatch for per-frame profiling."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.stats = defaultdict(float)

    def lap(self, name):
        now = time.perf_counter()
        self.stats[name] += (now - self.last) * 1000  # ms
        self.last = now

    def stop(self):
        total = (time.perf_counter() - self.t0) * 1000
        self.stats["TOTAL"] = total
        return dict(self.stats)

def _check_mask_area(x, y, r, contour, h, w):
    if r <= 0 or not (0 <= x < w and 0 <= y < h):
        return 0.0

    x, y, r = int(x), int(y), int(r)
    circle_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(circle_mask, (x, y), r, 255, -1)

    contour_mask = np.zeros_like(circle_mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    overlap = cv2.bitwise_and(circle_mask, contour_mask)
    return cv2.countNonZero(overlap) / cv2.countNonZero(circle_mask)

def canny_hough(frame, circle_queue):
    circle_queue.put(detect_circle(frame))

def hsv_mask(frame, hsv_queue, color_name):
    hsv_queue.put((find_blob(frame, [(COLORS[color_name]['lower'], COLORS[color_name]['higher'])]), color_name))

def fuse_detections(hsv_queue, circle_queue, h, w):
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
            if not matched and area > MIN_AREA:
                x_post, y_post, w_box, h_box = cv2.boundingRect(cnt)
                bboxes.append(((x_post, y_post, w_box, h_box), color))

    return bboxes

stream = Frame_Grabber(1)
stream.start()

frame_count = 0  # for throttling profiler output
frame_idx = 0
trackers = []
id_pool = set(range(1, MAX_TRACKERS + 1))

hsv_queue = queue.Queue(maxsize=4)
circle_queue = queue.Queue(maxsize=1)

while True:
    timer = StageTimer()  # start per-frame timer

    frame = stream.read_data()
    timer.lap("GRAB")
    if frame is None or frame.size == 0:
        continue

    frame = cv2.resize(frame, FRAME_SIZE)
    timer.lap("RESIZE")
    h, w = frame.shape[:2]

    t2 = threading.Thread(target=canny_hough, args=(frame, circle_queue))
    t2.start()
    threads = []
    hsv_img = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    for color_name in COLORS:
        t = threading.Thread(target=hsv_mask, args=(hsv_img, hsv_queue, color_name,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    t2.join()
    timer.lap("DETECT")

    detections = fuse_detections(hsv_queue, circle_queue, h, w)
    timer.lap("FUSE")
    mt, un_t, un_d = assign(trackers, [d[0] for d in detections])

    # update matched trackers
    for r, c in mt:
        trackers[r].update_loss(frame, detections[c][0], detections[c][1])
        trackers[r].misses = 0

    # increment miss counters on unmatched trackers
    for r in un_t:
        trackers[r].misses += 1

    # add new trackers if space allows
    num_live = len(trackers) - len(un_t)
    space_left = min(len(detections) - num_live, MAX_TRACKERS - len(trackers))
    for c in un_d:
        if space_left <= 0 or not id_pool:
            break
        new_id = id_pool.pop()
        bbox, color = detections[c]
        trackers.append(Kalman_tracker(new_id, bbox, frame, color))
        space_left -= 1

    # prune dead trackers
    alive_trackers = []
    for t in trackers:
        if t.misses < MAX_MISSES:
            alive_trackers.append(t)
        else:
            id_pool.add(t.id)
    trackers = alive_trackers

    # tracker update & drawing
    for t in trackers:
        if frame_idx:
            (cx, cy), ok = t.update(frame)
            
        else:
            cx, cy = t.predict_center()
            ok = False
        if frame_idx:
            x, y, w_box, h_box = t.bbox
            colour = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), colour, 2)
            cv2.putText(frame, f"Color: {t.color}, ID {t.id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
            cv2.imshow("Multi-Ball Tracker", frame)

    timer.lap("TRACK_DRAW")
    

    key = cv2.waitKey(1) & 0xFF
    stats = timer.stop()
    frame_count += 1  
    if frame_count % 30 == 0:
        print(f"{stats['TOTAL']:.1f} ms | " + " ".join(f"{k}:{v:.1f}" for k, v in stats.items() if k != 'TOTAL'))
    if key == ord('q'):
        break
    frame_idx ^=1
stream.stop()
cv2.destroyAllWindows()
