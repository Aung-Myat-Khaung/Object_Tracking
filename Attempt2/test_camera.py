import cv2
import time

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Try 1 or 2 if 0 doesn't work

if not cap.isOpened():
    print("❌ Failed to open camera.")
    exit()

print("✅ Camera opened successfully!")

# Warm-up delay
time.sleep(2)

ret, frame = cap.read()
if not ret:
    print("❌ Failed to grab frame.")
    cap.release()
    exit()

print("✅ Frame grabbed. Showing window...")

cv2.imshow("Camera Frame", frame)
cv2.waitKey(0)  # Wait until you press a key

cap.release()
cv2.destroyAllWindows()
