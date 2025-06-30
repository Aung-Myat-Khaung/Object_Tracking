from create_tracker import CreateTracker
from kalman import KalmanFilter
from config import TRACKER_TYPE
class Ball_Tracker():
    def __init__(self,frame,bbox):
        self.tracker = CreateTracker()
        self.tracker = self.tracker.create()
        self.tracker.init(frame,bbox)

        x,y,w,h = bbox
        self.kf = KalmanFilter(x+w/2,y+h/2)
        self.misses = 0
        self.bbox = bbox

    def predict(self):
        self.prediction = self.kf.predict()
        return self.prediction[0], self.prediction[1]

    def update(self, frame):
        ok, box = self.tracker.update(frame)
        if ok:
            x, y, w, h = box
            dx, dy = x + w/2, y + h/2
            self.kf.correct(dx, dy)
            self.misses = 0
            return True, (dx, dy, max(w,h)/2)
        # tracker failed
        dx, dy = map(int,self.kf.predict())
        self.misses += 1
        return False, (dx, dy, 0)

