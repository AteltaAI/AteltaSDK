# Utility functions for streaming videos using mediapipe cv2 and graph matching and 
# other bunch of utility functions like (counting some pose, correctness etc) 

import cv2 
import math 
import json 
import numpy as np
import mediapipe as mp 
import matplotlib.pyplot as plt 

from fastdtw import fastdtw
from scipy.spatial import procrustes
from scipy.spatial.distance import euclidean

from typing import List, Tuple, Optional, NamedTuple, Dict, Union 

# mediapipe libraries

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


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


def convert_pose_landmark_to_list(pose_results: Union[NamedTuple, None]) -> Union[List[List[float]], None]:
    """Converts medapipe results (results.pose_landmarks.landmark) to a list

    Args:
        pose_results (results.pose_landmarks.landmark): Mediapipe results

    Returns:
        _List[float] : List of floats of that results containing the (x, y, z) coordinates
    """

    if pose_results:
        landmarks = []
        for landmark in pose_results.pose_landmarks.landmark:
            landmarks.append((landmark.x, landmark.y, landmark.z))
        return landmarks
    return None 


def get_angle(keypoints: Union[np.ndarray, NamedTuple]) -> np.ndarray:
    """
    Get a list of angles for all the joints.
    args:
        keypoint: (Union[NamedTuple, np.ndarray]) Mediapipe keypoints
    """
    if type(keypoints) == NamedTuple:
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


def convert_mediapipe_pose_results_to_json(results : NamedTuple) -> Union[Dict[int, List[float]], None]:
    
    if results.pose_landmarks.landmark is not None:
        results_json = {} 
        for i, landmark in enumerate(results.pose_landmarks.landmark):
            results_json[i] = [landmark.x, landmark.y, landmark.z]
        
        return results_json
    else:
        return None 