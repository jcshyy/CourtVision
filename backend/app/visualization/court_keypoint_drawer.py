import cv2


class CourtKeypointDrawer:
    """Draws court keypoints with the same dots/labels as the reference drawer."""

    def __init__(self, min_keypoint_confidence=0.35):
        self.keypoint_color = (44, 44, 255)
        self.min_keypoint_confidence = min_keypoint_confidence

    def draw(self, frames, court_keypoints):
        output_frames = []

        for index, frame in enumerate(frames):
            annotated_frame = frame.copy()
            points = court_keypoints[index].xy.cpu().tolist()
            points = points[0] if points else []
            confidences = _keypoint_confidences(court_keypoints[index], len(points))

            for keypoint_index, point in enumerate(points):
                x, y = int(point[0]), int(point[1])
                if (
                    x <= 0
                    or y <= 0
                    or confidences[keypoint_index] < self.min_keypoint_confidence
                ):
                    continue

                cv2.circle(annotated_frame, (x, y), 8, self.keypoint_color, -1)
                cv2.putText(
                    annotated_frame,
                    str(keypoint_index),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            output_frames.append(annotated_frame)

        return output_frames


def _keypoint_confidences(frame_keypoints, point_count):
    confidences = [1.0] * point_count
    confidence_tensor = getattr(frame_keypoints, "conf", None)
    if confidence_tensor is None:
        return confidences

    confidence_values = confidence_tensor.cpu().tolist()
    if not confidence_values:
        return confidences

    confidence_values = confidence_values[0]
    for index, confidence in enumerate(confidence_values[:point_count]):
        confidences[index] = float(confidence)

    return confidences
