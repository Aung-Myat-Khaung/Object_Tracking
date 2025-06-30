import cv2
import numpy as np

class Kalman_filter():
    def __init__(self):
        self.kf = cv2.KalmanFilter(4,2) #(4,2) 4--> (position,velocity)
        self.kf.transitionMatrix = np.array([               # (value = pos x + pos y + vx + vy) 
                                        [1, 0, 1, 0],  # x = x + vx 
                                        [0, 1, 0, 1],  # y = y + vy
                                        [0, 0, 1, 0],  # vx = vx
                                        [0, 0, 0, 1]   # vy = vy
                                        ], dtype=np.float32)
        
        self.kf.measurementMatrix = np.array([             ## (value = pos x + pos y + vx + vy)
                                        [1, 0, 0, 0], #pos x
                                        [0, 1, 1, 1]  #pos y
                                        ], dtype=np.float32)
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.3 #larger value means the object moves erratically
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1 #smaller value means you trust the measurements more
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1 #it is about guessing the original postion --> smaller mean certain and larger mean not sure

    def initialize(self,x,y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)

    def correction(self, x, y):
        self.measurement = np.array([[x],[y]],np.float32)
        self.kf.correct(measurement=self.measurement)

    def prediction(self):
        self.predict = self.kf.predict()
        pred_x, pred_y = float(self.predict[0]),float(self.predict[1])
        return pred_x,pred_y