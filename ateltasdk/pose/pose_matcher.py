import cv2
import numpy as np
import mediapipe as mp

from fastdtw import fastdtw
from scipy.spatial import procrustes
from scipy.spatial.distance import euclidean
from typing import List, Tuple, Optional, Union, NamedTuple

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# This class implements matching of two static or dynamic poses.
# NOTE: This whole module explicitly depends on the conventions used in mediapipe.


def pose_matcher(
    angle_tracker1: np.ndarray,
    angle_tracker2: np.ndarray,
    evaluation_threshold: float,
    use_procustes: Optional[bool] = False,
) -> Tuple[float, bool]:
    """Matches two static or dynamic poses of two different poses w.r.t their change in angles and returns a score

    NOTE: Both angle_tracker1 and angle_tracker2 should be a matrix of size (n x 10) where n is the window size.
    taking n = 1 means the window size is 1 which is very instantaneous pose matching and considered as a static pose
    where taking a window size makes it more dynamic or more generalised.

    Args:
        angle_tracker1 (np.ndarray): The first pose angles
        angle_tracker2 (np.ndarray): The second pose angles
        evaluation_threshold (float): The threshold for evaluating the match between both the poses
        use_procustes (Optional[bool], optional): Whether to use procuest. Defaults to False.

    Returns:
        Tuple[float, bool]: The ecludian distance as score and whether they are a match or not
    """

    if use_procustes:
        _, angle_tracker2, _ = procrustes(
            angle_tracker1, angle_tracker2
        )  # not sure, might require more inspection

    distance, _ = fastdtw(angle_tracker1, angle_tracker2, dist=euclidean)

    return (distance, True if distance < evaluation_threshold else False)
