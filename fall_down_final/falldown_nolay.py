import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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

# [변경/추가] 주요 키포인트의 프레임간 이동 거리로 움직임(무동작) 판단
def check_immobility(current_landmarks, prev_landmarks, threshold=0.015):
    """
    주요 관절(어깨, 골반, 무릎, 손목 등)의 이동 거리를 계산하여 무동작 여부 반환
    threshold: 이 값보다 작게 움직이면 '정지'로 판단 (미세 미동 무시)
    """
    if prev_landmarks is None or current_landmarks is None:
        return False
        
    # 노이즈에 강하게 대응하기 위해 핵심 키포인트 8개 선정 (양 어깨, 골반, 무릎, 손목)
    check_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value
    ]
    
    total_movement = 0
    valid_points = 0
    
    for idx in check_indices:
        kp_curr = current_landmarks.landmark[idx]
        kp_prev = prev_landmarks.landmark[idx]
        
        # 신뢰도가 너무 낮은 포인트는 제외
        if kp_curr.visibility < 0.4 or kp_prev.visibility < 0.4:
            continue
            
        # 2D 평면상의 이동 거리 계산
        dist = np.sqrt((kp_curr.x - kp_prev.x)**2 + (kp_curr.y - kp_prev.y)**2)
        total_movement += dist
        valid_points += 1
        
    if valid_points == 0:
        return False
        
    avg_movement = total_movement / valid_points
    # 평균 이동 거리가 기준치(threshold)보다 작으면 '움직임 없음(True)'
    return avg_movement < threshold

cap = cv2.VideoCapture(0)
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
IMMOBILE_THRES = 0.008  # 미세 미동 차단 임계값 (상황에 따라 0.005 ~ 0.015 조절)

angle_history = deque()
prev_pose_landmarks = None  # 이전 프레임 저장용

fall_confirmed  = False
fall_duration   = None
immobile_since  = None      # [변경] 누운 기준이 아니라 '정지하기 시작한 시점' 추적
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
                        immobile_since = current_time
                    flashing = True
                else:
                    if not is_lying:
                        fall_confirmed = False
                        fall_duration  = None
                        flashing       = False
                        mask_visible   = False
                        immobile_since = None
                        help_triggered = False
                        angle_history.clear()

                # [변경] 낙상 확정 상태에서 '움직임 없음' 판단 로직
                if fall_confirmed:
                    is_immobile = check_immobility(results.pose_landmarks, prev_pose_landmarks, threshold=IMMOBILE_THRES)

                    if is_immobile:
                        if immobile_since is None:
                            immobile_since = current_time
                        immobile_duration = current_time - immobile_since
                        if immobile_duration >= HELP_SEC:
                            help_triggered = True
                    else:
                        # 움찔하거나 크게 움직이면 타이머 리셋
                        immobile_since = current_time

                # 다음 프레임을 위해 현재 랜드마크 저장
                prev_pose_landmarks = results.pose_landmarks

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
                    if immobile_since is not None:
                        immobile_duration = current_time - immobile_since
                        remaining = HELP_SEC - immobile_duration
                        # 정지 상태일 때만 카운트다운이 정상적으로 줄어듦
                        cv2.putText(image, f'HELP until: {max(0.0, remaining):.1f}s', (50, 240),
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
        else:
            # 사람 인식이 안 되더라도 직전 랜드마크는 초기화
            prev_pose_landmarks = None

        cv2.imshow('Fall Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()