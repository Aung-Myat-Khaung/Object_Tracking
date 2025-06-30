import cv2
from Color.COLOR import COLORS
from mask import get_mask
from km_filter import Kalman_filter


kalman_filter = []
multi_tracker = cv2.legacy.MultiTracker_create()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not working")


while True:
    ret, frame = cap.read()
    if ret is None:
        print("Cannot fetch frame")

    blurred = cv2.GaussianBlur(frame,(7,7),sigmaX=0)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask = get_mask(hsv,[(COLORS['green']['lower-hsv'],COLORS['green']['higher-hsv'])])
    contour, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    new_detection = []
    for cnt in contour:
        area = cv2.contourArea(cnt)
        if area >300:
            x, y, w, h = cv2.boundingRect(cnt)
            new_detection.append((x,y,w,h))

        for bbox in new_detection:
            tracker = cv2.legacy.TrackerCSRT()
            multi_tracker.add(tracker, frame, bbox)
            kf = Kalman_filter()
            kf.initialize((bbox[0]+bbox[2]) / 2, (bbox[1]+bbox[3]) / 2)
            kalman_filter.append(kf)
    ok, boxes = multi_tracker.update(frame)
    any_success = False
    for i, (x,y,w,h) in enumerate(boxes):
        cx, cy = int(x+w/2), int(y+h/2)
        if ok:  # MultiTracker returns global OK; we treat each box
            kalman_filter[i].correction(cx, cy)
            any_success = True
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        else:
            # if fail, rely on prediction
            cx, cy = map(int, kalman_filter[i].prediction())
            cv2.circle(frame, (cx, cy), 10, (0,0,255), 1)
    cv2.imshow("MultiTracker CSRT + Kalman", frame)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break
cap.release()
cv2.destroyAllWindows()