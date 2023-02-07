import cv2
import numpy as np
import mediapipe as mp
from typing import List, Optional, Union, Tuple
from mediapipe.python.solution_base import SolutionOutputs

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# TODO: Run different streamer in different processes

class SingleSourceStreamer:
    def __init__(self, source: Union[str, int]):
        self._cap = cv2.VideoCapture(source)

    @property
    def test_stream(self) -> None:
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        self._cap.release()
        cv2.destroyAllWindows()

    @property
    def yield_just_frames(self) -> Union[np.ndarray, None]:
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                yield frame
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    return None
            return None
        self._cap.release()

    def yield_frames_with_mediapipe(
        self,
        force_black_canvas: Optional[bool] = False,
        model_confidence: Optional[Union[str, int]] = 2,
        min_detection_confidence: Optional[bool] = 0.3,
        min_tracking_confidence: Optional[bool] = 0.28) -> Union[Tuple[np.ndarray, SolutionOutputs], Tuple[None, None]]:

        with mp_pose.Pose(
            model_complexity=model_confidence,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence) as pose:

            while self._cap.isOpened():
                ret, frame = self._cap.read()
