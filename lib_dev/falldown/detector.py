import cv2
import mediapipe as mp
import numpy as np
import time
import threading

from collections import deque

from config import *

from gemini_ai import GeminiAnalyzer

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class FallDetector:

    def __init__(self):

        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.angle_history = deque()

        self.movement_history = deque(maxlen=5)

        self.prev_pose_landmarks = None

        self.fall_confirmed = False
        self.fall_duration = 0

        self.immobile_since = None

        self.help_triggered = False

        self.gemini_analyzed = False

        self.flashing = False

        self.last_flash_time = 0

        self.flash_interval = 0.5

        self.mask_visible = False

        self.gemini = GeminiAnalyzer(
            GEMINI_API_KEY
        )

    def get_body_state(self, keypoints):

        try:

            ls = keypoints[
                mp_pose.PoseLandmark.LEFT_SHOULDER.value
            ]

            rs = keypoints[
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            ]

            lh = keypoints[
                mp_pose.PoseLandmark.LEFT_HIP.value
            ]

            rh = keypoints[
                mp_pose.PoseLandmark.RIGHT_HIP.value
            ]

            if any(
                kp.visibility < 0.2
                for kp in [ls, rs, lh, rh]
            ):
                return None, None

            mid_shoulder = np.array([
                (ls.x + rs.x) / 2,
                (ls.y + rs.y) / 2
            ])

            mid_hip = np.array([
                (lh.x + rh.x) / 2,
                (lh.y + rh.y) / 2
            ])

            spine_vec = mid_hip - mid_shoulder

            spine_angle = abs(np.degrees(
                np.arctan2(
                    abs(spine_vec[1]),
                    abs(spine_vec[0]) + 1e-6
                )
            ))

            all_x = [keypoints[i].x for i in range(33)]
            all_y = [keypoints[i].y for i in range(33)]

            aspect_ratio = (
                (max(all_x) - min(all_x)) /
                (max(all_y) - min(all_y) + 1e-6)
            )

            return spine_angle, aspect_ratio

        except:
            return None, None

    def check_immobility(self, current_landmarks):

        if (
            self.prev_pose_landmarks is None or
            current_landmarks is None
        ):
            return False

        check_indices = [
            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
            mp_pose.PoseLandmark.LEFT_HIP.value,
            mp_pose.PoseLandmark.RIGHT_HIP.value,
            mp_pose.PoseLandmark.LEFT_KNEE.value,
            mp_pose.PoseLandmark.RIGHT_KNEE.value
        ]

        total_movement = 0
        valid_points = 0

        for idx in check_indices:

            kp_curr = current_landmarks.landmark[idx]
            kp_prev = self.prev_pose_landmarks.landmark[idx]

            if (
                kp_curr.visibility < 0.2 or
                kp_prev.visibility < 0.2
            ):
                continue

            dist = np.sqrt(
                (kp_curr.x - kp_prev.x) ** 2 +
                (kp_curr.y - kp_prev.y) ** 2
            )

            total_movement += dist
            valid_points += 1

        if valid_points == 0:
            return False

        avg_movement = (
            total_movement / valid_points
        )

        self.movement_history.append(avg_movement)

        smoothed = (
            sum(self.movement_history) /
            len(self.movement_history)
        )

        return smoothed < IMMOBILE_THRES

    def reset_state(self):

        self.fall_confirmed = False
        self.fall_duration = 0
        self.flashing = False
        self.mask_visible = False
        self.immobile_since = None
        self.help_triggered = False
        self.gemini_analyzed = False

        self.angle_history.clear()
        self.movement_history.clear()

    def process(self, frame):

        image = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.pose.process(rgb)

        current_time = time.time()

        status = {
            "fall": self.fall_confirmed,
            "help": self.help_triggered,
            "spine": 0,
            "ratio": 0,
            "speed": 0,
            "fall_duration": self.fall_duration,
            "remaining": 0,
            "time": current_time
        }

        if results.pose_landmarks:

            keypoints = results.pose_landmarks.landmark

            spine_angle, aspect_ratio = (
                self.get_body_state(keypoints)
            )

            if spine_angle is not None:

                self.angle_history.append(
                    (current_time, spine_angle)
                )

                while (
                    self.angle_history and
                    current_time -
                    self.angle_history[0][0]
                    > HISTORY_SEC
                ):
                    self.angle_history.popleft()

                is_lying = (
                    spine_angle < FALL_ANGLE or
                    aspect_ratio > 1.3
                )

                angle_change_rate = 0

                if len(self.angle_history) >= 2:

                    oldest_t, oldest_a = (
                        self.angle_history[0]
                    )

                    dt = current_time - oldest_t

                    if dt > 0:

                        angle_change_rate = (
                            oldest_a - spine_angle
                        ) / dt

                is_fall = (
                    is_lying and
                    angle_change_rate > FALL_SPEED
                )

                if is_fall:

                    if not self.fall_confirmed:

                        self.fall_confirmed = True

                        self.fall_duration = (
                            current_time -
                            self.angle_history[0][0]
                        )

                        self.immobile_since = current_time

                    self.flashing = True

                else:

                    if not is_lying:
                        self.reset_state()

                if self.fall_confirmed:

                    is_immobile = (
                        self.check_immobility(
                            results.pose_landmarks
                        )
                    )

                    if is_immobile:

                        if self.immobile_since is None:
                            self.immobile_since = current_time

                        immobile_duration = (
                            current_time -
                            self.immobile_since
                        )

                        if immobile_duration >= HELP_SEC:

                            self.help_triggered = True

                            if not self.gemini_analyzed:

                                self.gemini_analyzed = True

                                t = threading.Thread(
                                    target=self.gemini.analyze,
                                    args=(image.copy(),)
                                )

                                t.start()

                    else:

                        self.immobile_since = current_time

                remaining = 0

                if self.immobile_since is not None:

                    immobile_duration = (
                        current_time -
                        self.immobile_since
                    )

                    remaining = max(
                        0.0,
                        HELP_SEC - immobile_duration
                    )

                status = {
                    "fall": self.fall_confirmed,
                    "help": self.help_triggered,
                    "spine": spine_angle,
                    "ratio": aspect_ratio,
                    "speed": angle_change_rate,
                    "fall_duration": self.fall_duration,
                    "remaining": remaining,
                    "time": current_time
                }

            self.prev_pose_landmarks = (
                results.pose_landmarks
            )

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        else:

            if (
                self.fall_confirmed and
                self.immobile_since is not None
            ):

                immobile_duration = (
                    current_time -
                    self.immobile_since
                )

                if (
                    immobile_duration >= HELP_SEC and
                    not self.gemini_analyzed
                ):

                    self.help_triggered = True
                    self.gemini_analyzed = True

                    t = threading.Thread(
                        target=self.gemini.analyze,
                        args=(image.copy(),)
                    )

                    t.start()

            self.prev_pose_landmarks = None

        if (
            self.flashing and
            not self.help_triggered
        ):

            if (
                current_time -
                self.last_flash_time
                >= self.flash_interval
            ):

                self.mask_visible = (
                    not self.mask_visible
                )

                self.last_flash_time = current_time

            if self.mask_visible:

                overlay = image.copy()

                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (
                        image.shape[1],
                        image.shape[0]
                    ),
                    (0, 0, 255),
                    -1
                )

                image = cv2.addWeighted(
                    overlay,
                    0.4,
                    image,
                    0.6,
                    0
                )

        return image, status
