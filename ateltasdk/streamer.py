import cv2
import numpy as np
import multiprocessing
import mediapipe as mp
from queue import Queue
from typing import List, Optional, Union, Tuple, NamedTuple, Dict, Any 

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

from ateltasdk.pose.pose_utils import (
    convert_pose_landmark_to_list, 
    convert_mediapipe_pose_results_to_json
)

from ateltasdk.pose.draw import PoseDraw 

# TODO: Run different streamer in different processes

class SingleSourceStreamer(PoseDraw):
    def __init__(self, source: Union[str, int]):
        self._cap = cv2.VideoCapture(source)
        super(SingleSourceStreamer, self).__init__() 

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
        draw : Optional[bool] = False, 
        force_black_canvas: Optional[bool] = False,
        model_confidence: Optional[Union[str, int]] = 2,
        min_detection_confidence: Optional[bool] = 0.3,
        min_tracking_confidence: Optional[bool] = 0.28) -> Union[Tuple[np.ndarray, NamedTuple], Tuple[np.ndarray, None]]:

        with mp_pose.Pose(
            model_complexity=model_confidence,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence) as pose:

            while self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if force_black_canvas:
                        canvas = np.zeros_like(frame)
                    else:
                        canvas = frame

                    frame.flags.writeable = False 
                    results = pose.process(frame)

                    if results is None:
                        yield (canvas, None)
                    
                    else:
                        canvas.flags.writeable = True 
                        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
                        if draw:
                            canvas = self.draw_mediapipe_pose(canvas, results)

                        yield (canvas, results)
    @property
    def close(self):
        self._cap.release() 
        cv2.destroyAllWindows() 


# Under developement. Requires Bug fixes.
class TwoSourceStreamer:
    def __init__(self, source1 : Union[str, int], source2 : Union[str, int]) -> None: 
        self.streamer1, self.streamer2 = SingleSourceStreamer(source1), SingleSourceStreamer(source2)
        self.Queue1, self.Queue2 = Queue(), Queue()
    
    # try to make it as a classmethod if possible 
    def pass_frames(self, streamer, queue, pass_results_as: Optional[str] = 'json', frame_name: Optional[str] = None, draw_keypoints : Optional[bool] = False, force_black_canvas : Optional[bool] = False):
        for (image, results) in streamer.yield_frames_with_mediapipe(draw=draw_keypoints, force_black_canvas=force_black_canvas):
            if results is not None:
                if pass_results_as == 'json' : results_to_pass = convert_mediapipe_pose_results_to_json(results)
                else:
                    results_to_pass = convert_pose_landmark_to_list(results)
            else:
                print("ohh")
                results_to_pass = None 
            queue.put((image, results_to_pass))

            if frame_name:
                cv2.imshow(frame_name, image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break 
        streamer.close 
    
    def concatinate_incoming_frames(self, frame_name: Optional[str] = None):
        while True:
            if (not self.Queue1.empty()) and (not self.Queue2.empty()):
                frame1, result1 = self.Queue1.get() 
                frame2, result2 = self.Queue2.get()
                print((frame1.shape, type(result1)), (frame2.shape, type(result2)))

    # work on this function more 
    def init_process(self, parameters: Tuple[Any], concatinated_frame_name: Optional[str] = None):
        """Does the two streaming processes in parallel
        
        This is how parameters should be initialized:
        ([pass_results_as, frame_name, draw_keypoints, force_black_canvas], [pass_results_as, frame_name, draw_keypoints, force_black_canvas])
        Args:
            parameters (Tuple[Any]): _description_
        """
        pass_results_as_for_source1, frame_name_source1, draw_keypoints_for_source1, force_black_canvas_for_source1 = parameters[0]
        pass_results_as_for_source2, frame_name_source2, draw_keypoints_for_source2, force_black_canvas_for_source2 = parameters[1]

        process1 = multiprocessing.Process(target=self.pass_frames, args=(
            self.streamer1, self.Queue1, pass_results_as_for_source1, frame_name_source1, draw_keypoints_for_source1, force_black_canvas_for_source1
        ))

        process2 = multiprocessing.Process(target=self.pass_frames, args=(
            self.streamer2, self.Queue2, pass_results_as_for_source2, frame_name_source2, draw_keypoints_for_source2, force_black_canvas_for_source2
        ))

        process3 = multiprocessing.Process(target=self.concatinate_incoming_frames, args=(concatinated_frame_name))
        return process1, process2, process3



if __name__ == "__main__":
    path = '/home/anindya/Documents/Atelta/AteltaStream/.data/raw/Waterfall.mp4'
    source1, source2 = 0, path
    parameters = (
        ['list', 'Frame1', True, False],
        ['list', 'Frame2', True, False]
    )

    two_source_streamer = TwoSourceStreamer(source1, source2)
    process1, process2, process3 = two_source_streamer.init_process(parameters, concatinated_frame_name='Output')

    process1.start() 
    process2.start()
    #process3.start() 
    cv2.destroyAllWindows() 