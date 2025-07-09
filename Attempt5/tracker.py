import cv2
import numpy as np
from kalman import Kalman_filter

class Kalman_tracker:
    def __init__(self, id, bbox,frame,color):
        self.id = id
        self.tracker = cv2.TrackerKCF_create()
        self.bbox = bbox
        self.tracker.init(frame,self.bbox)
        x,y,w,h = self.bbox
        self.kf = Kalman_filter(x,y)
        self.misses = 0
        self.color = color

    def predict_center(self):
        return self.kf.predict()
    
    def update_loss(self,frame,box,color):
        self.tracker = cv2.TrackerKCF_create()
        x,y,w,h = box
        self.bbox = (x,y,w,h)
        self.tracker.init(frame,self.bbox)
        self.misses = 0
        self.color = color
        self.kf.correct(x+w/2,y+h/2)
    
    def update(self,frame):
        ok, bbox = self.tracker.update(frame)
        if ok:
            x,y,w,h = bbox
            center_x,center_y = x+w/2,y+h/2
            self.kf.correct(center_x,center_y)
            self.misses = 0
            return (int(center_x),int(center_y)),ok
        else:
            self.misses+=1
            predict_x,predict_y = self.kf.predict()
            return (int(predict_x),int(predict_y)),ok