import cv2
import numpy as np

class KalmanFilter:
    def __init__(self, x, y):

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix  = np.array([
                                            [1,0,1,0],
                                            [0,1,0,1],
                                            [0,0,1,0],
                                            [0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([
                                            [1,0,0,0],
                                            [0,1,0,0]], np.float32)
        self.kf.processNoiseCov   = np.eye(4, dtype=np.float32) * 0.15
        self.kf.statePost         = np.array([[x],[y],[0],[0]], np.float32)

    def predict(self):
        p = self.kf.predict();  
        return p[0][0], p[1][0]
    
    def correct(self, x, y):
        self.kf.correct(np.array([[x],[y]], np.float32))
