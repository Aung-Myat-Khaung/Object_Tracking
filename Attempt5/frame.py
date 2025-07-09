import cv2
import threading
import time
class Frame_Grabber():
    def __init__(self,src = 0):
        self.cap = cv2.VideoCapture(src,cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.lock = threading.Lock()
        self.stopped = False
        self.frame = None
    
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
    
    def update(self):
        while not self.stopped:
            ret, f = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.frame = f.copy()
            time.sleep(0)
        
    def dimensions(self):
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        return self.frame_width,self.frame_height,self.fps
    
    def read_data(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()