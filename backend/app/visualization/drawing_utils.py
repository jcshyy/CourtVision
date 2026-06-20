import cv2
import numpy as np

from backend.app.utils.geometry import bbox_center, bbox_width


def draw_triangle(frame, bbox, color):
    x_center, y_top = _top_center(bbox)
    triangle_points = np.array(
        [
            [x_center, y_top],
            [x_center - 10, y_top - 20],
            [x_center + 10, y_top - 20],
        ],
        dtype=np.int32,
    )

    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)
    return frame


def draw_ellipse(frame, bbox, color, track_id=None):
    x_center, _ = bbox_center(bbox)
    y_bottom = int(bbox[3])
    width = bbox_width(bbox)

    cv2.ellipse(
        frame,
        center=(int(x_center), y_bottom),
        axes=(int(width), int(0.35 * width)),
        angle=0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4,
    )

    if track_id is not None:
        _draw_track_id(frame, x_center, y_bottom, track_id, color)

    return frame


def draw_possession_marker(frame, bbox, color=(0, 255, 255)):
    x_center, _ = bbox_center(bbox)
    y_top = int(bbox[1])
    radius = 7

    cv2.circle(
        frame,
        (int(x_center), max(0, y_top - 18)),
        radius,
        color,
        cv2.FILLED,
    )
    cv2.circle(
        frame,
        (int(x_center), max(0, y_top - 18)),
        radius,
        (0, 0, 0),
        2,
    )
    return frame


def _draw_track_id(frame, x_center, y_bottom, track_id, color):
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x_center - rectangle_width // 2
    x2_rect = x_center + rectangle_width // 2
    y1_rect = (y_bottom - rectangle_height // 2) + 15
    y2_rect = (y_bottom + rectangle_height // 2) + 15

    cv2.rectangle(
        frame,
        (int(x1_rect), int(y1_rect)),
        (int(x2_rect), int(y2_rect)),
        color,
        cv2.FILLED,
    )

    x1_text = x1_rect + 12
    if track_id > 99:
        x1_text -= 10

    cv2.putText(
        frame,
        f"{track_id}",
        (int(x1_text), int(y1_rect + 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
    )


def _top_center(bbox):
    x1, y1, x2, _ = bbox
    return int((x1 + x2) / 2), int(y1)
