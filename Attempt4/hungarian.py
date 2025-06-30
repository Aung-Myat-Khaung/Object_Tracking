import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import linalg
from config import MAX_DISTANCE

def bbox_center(box):

    x,y,w,h = box
    return np.array([x+w/2,y+h/2],np.float32)

def _build_cost_matrix(trackers, detections):
    cost = np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for row, t in enumerate(trackers):
        predict_c = t.predict_center()
        for column, box in enumerate(detections):
            detected_c = bbox_center(box)
            cost[row,column] = np.linalg.norm(predict_c-detected_c)
    return cost

def assign(trackers, detections):
    number_t, number_d = len(trackers),len(detections)
    if number_d==0 or number_t == 0:
        return [],list(range(number_t)),list(range(number_d))
    
    cost = _build_cost_matrix(trackers,detections)
    row_index, column_index = linear_sum_assignment(cost)
    matches, unmatched_trackrs, unmatched_detections =[], [], []

    unmatched_trackers = list(set(range(number_t))-set(row_index))
    unmatched_detections = list(set(range(number_d))-set(column_index))

    for row, column in zip(row_index,column_index):
        if cost[row,column] <= MAX_DISTANCE:
            matches.append((row,column))
        else:
            unmatched_trackers.append(row)
            unmatched_detections.append(column)
    return matches, unmatched_trackers, unmatched_detections