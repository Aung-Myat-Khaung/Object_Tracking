import cv2
import numpy as np
from kalman import Kalman_filter

class Kalman_tracker:
    def __init__ (self, idt, bbox, frame):
        self.id = idt
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame,bbox)
        x,y,w,h = bbox
        self.kf = Kalman_filter(x+w/2,y+h/2)
        self.bbox = bbox
        self.misses = 0

    def predict_center(self):
        return self.kf.predict()
    
    def update_detection(self, frame, bbox):
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame,bbox)
        x,y,w,h = bbox
        self.kf.correct(x+w/2,y+h/2)
        self.bbox = bbox
        self.misses = 0
    def update_post(self,frame):
        ok, bbox = self.tracker.update(frame)
        if ok:
            x,y,w,h = map(int,bbox)
            dx, dy = x+w/2,y+h/2
            self.kf.correct(dx,dy)
            self.misses = 0
            return(int(dx),int(dy)),ok
        else:
            self.misses +=1
            px,py = self.kf.predict()
            return (int(px),int(py)), ok