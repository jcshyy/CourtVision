from backend.app.visualization.drawing_utils import draw_triangle


class BallTracksDrawer:
    """Draws tracked basketball positions on video frames."""

    def __init__(self, color=(0, 255, 0)):
        self.color = color

    def draw(self, frames, tracked_results):
        output_frames = [frame.copy() for frame in frames]

        for frame, result in zip(output_frames, tracked_results):
            if isinstance(result, dict):
                for ball in result.values():
                    if ball.get("bbox") is not None:
                        draw_triangle(frame, ball["bbox"], self.color)
                continue

            if _is_bbox(result):
                draw_triangle(frame, result, self.color)
                continue

            if result is None:
                continue

            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            bbox = _highest_confidence_bbox(boxes)
            draw_triangle(frame, bbox, self.color)

        return output_frames


def _is_bbox(value):
    return isinstance(value, (list, tuple)) and len(value) == 4


def _highest_confidence_bbox(boxes):
    bboxes = boxes.xyxy.cpu().tolist()
    if boxes.conf is None:
        return bboxes[0]

    confidences = boxes.conf.cpu().tolist()
    best_index = max(range(len(confidences)), key=confidences.__getitem__)
    return bboxes[best_index]
