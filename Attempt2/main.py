import cv2
from detector import ColorDetection
from tracker import Ball_Tracker
from create_tracker import CreateTracker
import numpy as np

from config import MAX_MISS, TRACKER_TYPE, COLORS, MAX_DIST

detectors = ColorDetection([(COLORS['green']['lower-hsv'],COLORS['green']['higher-hsv'])],'green')
trackers = []
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Camera not opened")
    SystemExit

while True:
    ret, frame = cap.read()
    if ret is None:
        print("Cannot fetch frame.")
        SystemExit

    boxes = detectors.detections(frame)

    assign = [False]*len(boxes)
    for ball in trackers:
        ball_id, ball_dist = -1, MAX_DIST
        px, py = ball.predict()

        for i, (x, y, w, h) in enumerate(boxes):
            if assign[i]:
                continue
            dx, dy = (x + w) / 2, (y + h) / 2
            dist = np.hypot(px - dx, py - dy)
            if dist < ball_dist:
                ball_id, ball_dist = i, dist

        if ball_id != -1:
            bbox = boxes[ball_id]
            tracker_creator = CreateTracker()
            ball.tracker = tracker_creator.create()
            ball.tracker.init(frame, bbox)
            assign[ball_id] = True


    for i, used in enumerate(assign):
        if not used:
            trackers.append(Ball_Tracker(frame, boxes[i]))

    trackers = [t for t in trackers if (t.misses< MAX_MISS)]
    for t in trackers:
        ok, (cx, cy, r) = t.update(frame)
        color = (0,255,0) if ok else (0,0,255)
        cv2.circle(frame, (int(cx),int(cy)), int(max(r,8)), color, 2)
        cv2.putText(frame, f"ID {t.id}", (cx, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    cv2.imshow('Ball Tracking', frame)
    if cv2.waitKey(1)&0xFF==ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()
