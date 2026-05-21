import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def get_body_state(keypoints):
    try:
        ls = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]

        if any(kp.visibility < 0.4 for kp in [ls, rs, lh, rh]):
            return None, None

        mid_shoulder = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
        mid_hip      = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])

        spine_vec   = mid_hip - mid_shoulder
        spine_angle = abs(np.degrees(np.arctan2(abs(spine_vec[1]),
                                                 abs(spine_vec[0]) + 1e-6)))

        all_x = [keypoints[i].x for i in range(33)]
        all_y = [keypoints[i].y for i in range(33)]
        aspect_ratio = (max(all_x) - min(all_x)) / (max(all_y) - min(all_y) + 1e-6)

        return spine_angle, aspect_ratio
    except:
        return None, None

def is_lying_down(landmarks):
    try:
        lm = landmarks.landmark

        nose       = lm[mp_pose.PoseLandmark.NOSE]
        l_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip      = lm[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip      = lm[mp_pose.PoseLandmark.RIGHT_HIP]
        l_ankle    = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle    = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

        mid_shoulder = np.array([(l_shoulder.x + r_shoulder.x) / 2,
                                  (l_shoulder.y + r_shoulder.y) / 2])
        mid_hip      = np.array([(l_hip.x + r_hip.x) / 2,
                                  (l_hip.y + r_hip.y) / 2])
        spine_vec    = mid_hip - mid_shoulder
        spine_angle  = abs(np.degrees(np.arctan2(abs(spine_vec[1]),
                                                  abs(spine_vec[0]) + 1e-6)))

        all_x = [lm[i].x for i in range(33)]
        all_y = [lm[i].y for i in range(33)]
        aspect_ratio = (max(all_x) - min(all_x)) / (max(all_y) - min(all_y) + 1e-6)

        mid_ankle     = np.array([(l_ankle.x + r_ankle.x) / 2,
                                   (l_ankle.y + r_ankle.y) / 2])
        vertical_span = abs(nose.y - mid_ankle[1])

        # 척추가 수평 AND (가로비율 OR 수직범위) 둘 다 만족해야 누운 것
        lying = (
            spine_angle < 35
            and (aspect_ratio > 1.3 or vertical_span < 0.25)
        )
        return lying
    except:
        return False


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

ret, frame = cap.read()
h, w = frame.shape[:2]
cv2.namedWindow('Fall Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fall Detection', w, h)

HISTORY_SEC = 1.0
FALL_SPEED  = 35.0
FALL_ANGLE  = 40.0
HELP_SEC    = 10.0

angle_history = deque()

fall_confirmed  = False
fall_duration   = None
lying_since     = None
help_triggered  = False

flashing        = False
last_flash_time = 0
flash_interval  = 0.5
mask_visible    = False

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image     = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = pose.process(image_rgb)

        current_time = time.time()

        if results.pose_landmarks:
            keypoints = results.pose_landmarks.landmark
            spine_angle, aspect_ratio = get_body_state(keypoints)

            if spine_angle is not None:
                angle_history.append((current_time, spine_angle))
                while angle_history and current_time - angle_history[0][0] > HISTORY_SEC:
                    angle_history.popleft()

                is_lying = (spine_angle < FALL_ANGLE or aspect_ratio > 1.3)

                angle_change_rate = 0.0
                if len(angle_history) >= 2:
                    oldest_t, oldest_a = angle_history[0]
                    dt = current_time - oldest_t
                    if dt > 0:
                        angle_change_rate = (oldest_a - spine_angle) / dt

                is_fall = is_lying and angle_change_rate > FALL_SPEED

                if is_fall:
                    if not fall_confirmed:
                        fall_confirmed = True
                        fall_duration  = current_time - angle_history[0][0]
                        lying_since    = current_time
                    flashing = True
                else:
                    if not is_lying:
                        fall_confirmed = False
                        fall_duration  = None
                        flashing       = False
                        mask_visible   = False
                        lying_since    = None
                        help_triggered = False
                        angle_history.clear()

                if fall_confirmed:
                    currently_lying = is_lying_down(results.pose_landmarks)

                    if currently_lying:
                        if lying_since is None:
                            lying_since = current_time
                        lying_duration = current_time - lying_since
                        if lying_duration >= HELP_SEC:
                            help_triggered = True
                    else:
                        lying_since = current_time

                # ── 화면 출력 ───────────────────────────────
                if help_triggered:
                    if int(current_time * 2) % 2 == 0:
                        cv2.putText(image, '!! HELP !!', (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 7)

                elif fall_confirmed:
                    cv2.putText(image, 'FALL DETECTED', (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
                    if fall_duration is not None:
                        cv2.putText(image, f'Fall Time: {fall_duration:.2f}s', (50, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 4)
                    if lying_since is not None:
                        lying_duration = current_time - lying_since
                        remaining = HELP_SEC - lying_duration
                        cv2.putText(image, f'HELP until: {remaining:.1f}s', (50, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)

                cv2.putText(image,
                            f'spine:{spine_angle:.1f}  ratio:{aspect_ratio:.2f}  speed:{angle_change_rate:.1f}',
                            (20, image.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

            if flashing and not help_triggered:
                if current_time - last_flash_time >= flash_interval:
                    mask_visible    = not mask_visible
                    last_flash_time = current_time
                if mask_visible:
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0, 0),
                                  (image.shape[1], image.shape[0]), (0, 0, 255), -1)
                    image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

            mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Fall Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()