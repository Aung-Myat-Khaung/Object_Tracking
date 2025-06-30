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

stop_event = threading.Event()
stream = Frame_Grabber(1)
stream.start()
def grab_frame(frame_queue):
    while not stop_event.is_set():
        raw = stream.read_data()  
        if raw is None or raw.size == 0:
            continue 
        raw = cv2.resize(raw,FRAME_SIZE)
        cv2.imshow('raw',raw)
        frame_queue.put(raw)
    frame_queue.put(None)

def detect_balls(frame_queue, detection_queue):
    while not stop_event.is_set():
        frame = frame_queue.get()
        cv2.imshow("Frame",frame)
        final_mask = find_blob(frame, [(COLORS['green']['lower'],COLORS['green']['higher'])])
        cv2.imshow('mask',final_mask)
        circles = detect_circle(final_mask)
        print(circles)
        detection_queue.put((frame,circles))

def predict_ball_movement(detection_queue):
    trackers = []
    tracker_id = 0

    while True:
        readings = detection_queue.get()
        if readings is None or stop_event.is_set():
            break
        frame, detection = readings
        
        p_center = [t.predict_center() for t in trackers]
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
            x,y,w,h = t.bbox
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, f"ID {t.id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imshow("Multi_Tracker",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            stop_event.set()
            
            break
        





if __name__=='__main__':
    frame_queue = queue.Queue(maxsize=5)
    detection_queue = queue.Queue(maxsize=5)


    t1 = threading.Thread(target=grab_frame, args=(frame_queue,))
    t2 = threading.Thread(target=detect_balls, args=(frame_queue, detection_queue))
    t3 = threading.Thread(target=predict_ball_movement, args=(detection_queue,))

    t1.start()
    time.sleep(0.001)
    t2.start()
    time.sleep(0.001)
    t3.start()

    # Wait for threads to finish
    t1.join()
    t2.join()
    t3.join()
    stream.stop()

    
    cv2.destroyAllWindows()