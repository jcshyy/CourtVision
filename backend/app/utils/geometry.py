from math import sqrt


def bbox_width(box):
    x1, _, x2, _ = box
    return x2 - x1


def bbox_height(box):
    _, y1, _, y2 = box
    return y2 - y1


def bbox_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def euclidean_distance(point_a, point_b):
    return sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)
