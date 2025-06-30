from main import color_classification
from COLOR import COLORS

def test_color_classification():
    for color, props in COLORS.items():
        result = color_classification(props["file_loc"]) 
        assert result == color,(
            f"\nInput:"
            f"\n  lower-hsv = {props['lower-hsv']!r}"
            f"\n  higher-hsv = {props['higher-hsv']!r}"
            f"\nExpected: {color!r}"
            f"\nGot:      {result!r}"
        )
            

    