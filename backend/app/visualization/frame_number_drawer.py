import cv2


class FrameNumberDrawer:
    def draw(self, frames):
        output_frames = []

        for index, frame in enumerate(frames):
            output_frame = frame.copy()
            cv2.putText(
                output_frame,
                str(index),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            output_frames.append(output_frame)

        return output_frames
