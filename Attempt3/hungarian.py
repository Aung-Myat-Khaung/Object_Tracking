import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import linalg


def bbox_center(box):

    x,y,w,h = box
    return np.array([x+w/2,y+h/2],np.float32)

def _build_cost_matrix(tracks, detections):
    cost = np.zeros((len(tracks),len(detections)),dtype=np.float32)
    for r, t in enumerate(tracks):
        pc = t.predict_center()  # (2,) float
        for c, box in enumerate(detections):
            dc = bbox_center(box)
            cost[r, c] = np.linalg.norm(pc - dc)
    return cost

def assign(tracks, detections, max_distance):
    n_t, n_d = len(tracks), len(detections)
    if n_t == 0 or n_d == 0:
        return [], list(range(n_t)), list(range(n_d))

    cost = _build_cost_matrix(tracks, detections)
    row_ind, col_ind = linear_sum_assignment(cost)

    matches, unmatched_tracks, unmatched_dets = [], [], []

    unmatched_tracks = list(set(range(n_t)) - set(row_ind))
    unmatched_dets = list(set(range(n_d)) - set(col_ind))

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= max_distance:
            matches.append((r, c))
        else:
            unmatched_tracks.append(r)
            unmatched_dets.append(c)

    return matches, unmatched_tracks, unmatched_dets