import cv2
import numpy as np


class Homography:
    def __init__(
        self,
        source: np.ndarray,
        target: np.ndarray,
        method: int = 0,
        ransac_reproj_threshold: float = 3.0,
    ) -> None:
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)

        self.matrix, self.mask = cv2.findHomography(
            source,
            target,
            method,
            ransac_reproj_threshold,
        )
        if self.matrix is None:
            raise ValueError("Homography matrix could not be calculated.")

        self.inlier_count = (
            int(self.mask.sum()) if self.mask is not None else int(source.shape[0])
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        points = points.reshape(-1, 1, 2).astype(np.float32)
        points = cv2.perspectiveTransform(points, self.matrix)
        return points.reshape(-1, 2).astype(np.float32)
