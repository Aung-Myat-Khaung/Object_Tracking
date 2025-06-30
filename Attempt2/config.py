COLORS = {
    "red": {
        "lower-hsv": (0, 50, 50),
        "higher-hsv": (4, 255, 255), 
        "file_loc" : 'test/color/red_orange.jpg'       
    },
    "green": {
        "lower-hsv": (25, 50, 50),
        "higher-hsv": (80, 255, 255),
        "file_loc" : 'test/color/green.jpg'   
    },
    "blue": {
        "lower-hsv": (100, 50, 50),
        "higher-hsv": (130, 255, 255),
        "file_loc" : 'test/color/blue.jpg'   
    },
    "yellow": {
        "lower-hsv": (22, 50, 50),
        "higher-hsv": (30, 255, 255),
        "file_loc" : 'test/color/yellow.jpg'   
    },
    "orange": {
        "lower-hsv": (7, 50, 50),
        "higher-hsv": (20, 255, 255),
        "file_loc" : 'test/color/orange.jpg'   

    },
    "purple": {
        "lower-hsv": (130, 50, 50),
        "higher-hsv": (160, 255, 255),
        "file_loc" : 'test/color/purple.jpg'   
    },
    "gray":
    {
        "lower-hsv": (0, 0, 20),
        "higher-hsv": (255, 25, 200)
    }
}

TRACKER_TYPE = 'KCF'
FRAME_SIZE = (640,480)
MAX_MISS = 70
MAX_DIST = 300