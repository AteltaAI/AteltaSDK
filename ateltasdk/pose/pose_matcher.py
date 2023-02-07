import cv2
import numpy as np
import mediapipe as mp

from fastdtw import fastdtw
from scipy.spatial import procrustes
from scipy.spatial.distance import euclidean
from typing import List, Tuple, Optional, Union
from mediapipe.python.solution_base import SolutionOutputs

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# This class implements matching of two static or dynamic poses.
# NOTE: This whole module explicitly depends on the conventions used in mediapipe.


def _find_angle(node_joint_triplet: Union[List[List[float]], np.ndarray]) -> float:
    """
    A trivial method which assumes that it will accept a array or list of list of shape (3 x 3)
    Where each row corresponds to a node (which is a part of a joint). For example the joint between
    hands and shoulder.

    args:
        node_joint_triplet : (Union[List[List[float]], np.ndarray]) A list of list or a numpy array of shape (3 x 3)

    returns:
        The angle between the joints
    """

    radians = np.arctan2(
        node_joint_triplet[2][1] - node_joint_triplet[1][1],
        node_joint_triplet[2][0] - node_joint_triplet[1][0],
    ) - np.arctan2(
        node_joint_triplet[0][1] - node_joint_triplet[1][1],
        node_joint_triplet[0][0] - node_joint_triplet[1][0],
    )

    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle


def convert_pose_landmark_to_list(pose_results: SolutionOutputs) -> List[List[float]]:
    """Converts medapipe results (results.pose_landmarks.landmark) to a list

    Args:
        pose_results (results.pose_landmarks.landmark): Mediapipe results

    Returns:
        _List[float] : List of floats of that results containing the (x, y, z) coordinates
    """

    landmarks = []
    for landmark in pose_results.pose_landmarks.landmark:
        landmarks.append((landmark.x, landmark.y))
    return landmarks


def get_angle(keypoints: Union[np.ndarray, SolutionOutputs]) -> np.ndarray:
    """
    Get a list of angles for all the joints.
    args:
        keypoint: (Union[SolutionOutputs, np.ndarray]) Mediapipe keypoints
    """
    if type(keypoints) == SolutionOutputs:
        keypoints = convert_pose_landmark_to_list(keypoints)

    # Conventions based on mediapipe
    keypoints_positions = [
        [16, 14, 12],
        [14, 12, 24],
        [12, 24, 26],
        [24, 26, 28],
        [26, 28, 32],
        [15, 13, 11],
        [13, 11, 23],
        [11, 23, 25],
        [23, 25, 27],
        [25, 27, 31],
    ]

    angles = []
    for pose in keypoints_positions:
        angles.append(_find_angle(keypoints[pose]))
    return np.array(angles) / np.linalg.norm(angles)


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
