import cv2
import numpy as np

# --- helper for the trackbars (OpenCV requires a callback) -------------------
def _noop(val: int) -> None:
    pass

# --- video source -----------------------------------------------------------
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)          # change index if you have multiple cameras

# --- UI for interactive tuning ---------------------------------------------
cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Canny Low",  "Controls",  50, 500, _noop)
cv2.createTrackbar("Canny High", "Controls", 150, 500, _noop)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (9, 9), 2)   # de-noise before Canny

    low  = cv2.getTrackbarPos("Canny Low",  "Controls")
    high = cv2.getTrackbarPos("Canny High", "Controls")

    edges = cv2.Canny(blur, low, high)

    # HoughCircles runs its own internal Canny; we pass `high` in via param1
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.0, 
        minDist=20,           # inverse accumulator resolution
        param1=70,        # high threshold for internal Canny
        param2=30,         # accumulator threshold (smaller → more detections)
        minRadius=20,
        maxRadius=50)       # 0 → no upper bound on radius

    if circles is not None:
        for x, y, r in np.uint16(np.around(circles))[0]:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)          # circle edge
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 3)          # center dot

    cv2.imshow("Edges",   edges)   # visualise Canny result
    cv2.imshow("Circles", frame)   # annotated live preview

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
