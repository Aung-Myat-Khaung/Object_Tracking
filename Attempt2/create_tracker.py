import cv2
from config import TRACKER_TYPE
class CreateTracker():
    def create(self):
        name = TRACKER_TYPE
        if name == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif name == 'KCF':
            return cv2.TrackerKCF_create()
        else:
            print("Type Error")
            SystemExit
            return

        