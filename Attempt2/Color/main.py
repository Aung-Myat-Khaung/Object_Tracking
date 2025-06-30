import cv2
from Color.util_hsv import get_HSVmask
from COLOR import COLORS

def color_classification(pic):
    img = cv2.imread(pic,cv2.IMREAD_COLOR)
    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for color, props in COLORS.items():
        color_range = [(props['lower-hsv'],props['higher-hsv'])]
        mask = get_HSVmask(hsv_img, color_range)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >0:
                return color
        