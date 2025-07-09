#!/usr/bin/env python3
"""
Show the live camera feed, draw every detected blob, and print each blob’s area
(in square pixels) on every frame, using the same HSV threshold(s) you already use.
"""

import cv2
import numpy as np

# ── 1.  DEFINE YOUR HSV THRESHOLD(S)  ──────────────────────────────────────────
# If you’re already importing a COLORS dict from config.py, just replace this
# block with:  from config import COLORS;  COLOR_RANGES = [COLORS['green']]
COLOR_RANGES = [              # list of dicts or 2-tuples: lower HSV, upper HSV
    {"lower": (25, 50, 50), "upper": (80, 255, 255)}   # example “green” band
]
# If you need multiple colours, append more dicts/tuples to COLOR_RANGES.


# ── 2.  VIDEO SOURCE  ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)   # 0 = first webcam; adjust if needed
if not cap.isOpened():
    raise SystemExit("❌  Couldn’t open camera - check the index or permissions")


# ── 3.  MAIN LOOP  ────────────────────────────────────────────────────────────
print("Press  q  to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame,(640,480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Build a mask that combines *all* colour ranges
    mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for rng in COLOR_RANGES:
        lower = np.array(rng["lower"], dtype=np.uint8)
        upper = np.array(rng["upper"], dtype=np.uint8)
        mask_total |= cv2.inRange(hsv, lower, upper)

    # Clean up speckle
    mask_total = cv2.erode(mask_total, None, 2)
    mask_total = cv2.dilate(mask_total, None, 2)

    # Find contours in the binary mask
    contours, _ = cv2.findContours(mask_total, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    areas_this_frame = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:                           # ignore tiny noise blobs
            continue

        areas_this_frame.append(area)

        # Draw the contour + numeric area on the frame
        cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 1)
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(frame, f"{int(area)}",
                    (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # ── 4.  PRINT AREAS TO CONSOLE  ──────────────────────────────────────────
    if areas_this_frame:
        print("Areas (px²):", ", ".join(f"{a:.0f}" for a in areas_this_frame))

    # ── 5.  SHOW WINDOWS  ────────────────────────────────────────────────────
    cv2.imshow("frame", frame)
    cv2.imshow("mask", mask_total)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ── 6.  CLEAN-UP  ────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
