import cv2
import mediapipe as mp
import math

# [기준값 정의]
STATE_THRESHOLD_VERTICAL = 30
STATE_THRESHOLD_HORIZONTAL = 60

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(1) # 연결된 카메라 번호에 맞게 수정 (0 또는 1)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    upper_state = "IDLE"
    lower_state = "IDLE"
    upper_angle = 0.0
    lower_angle = 0.0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            # 1. 상체 포인트 (어깨 중앙, 골반 중앙)
            shld_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
            shld_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
            hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2

            # 2. 하체 포인트 수정: 발목(ANKLE) -> 무릎(KNEE)
            knee_x = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x + landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x) / 2
            knee_y = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y + landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2

            # --- 상체 각도 계산 ---
            u_dx = (shld_x - hip_x) * w
            u_dy = (shld_y - hip_y) * h 
            upper_angle = abs(math.degrees(math.atan2(u_dx, u_dy)))
            if upper_angle > 90: upper_angle = abs(upper_angle - 180)

            # --- 하체 각도 계산 (골반-무릎 기준) ---
            l_dx = (hip_x - knee_x) * w
            l_dy = (hip_y - knee_y) * h
            lower_angle = abs(math.degrees(math.atan2(l_dx, l_dy)))
            if lower_angle > 90: lower_angle = abs(lower_angle - 180)

            # --- 개별 상태 판별 ---
            if upper_angle < STATE_THRESHOLD_VERTICAL: upper_state = "VERTICAL"
            elif upper_angle > STATE_THRESHOLD_HORIZONTAL: upper_state = "HORIZONTAL"
            else: upper_state = "SLANTED"

            if lower_angle < STATE_THRESHOLD_VERTICAL: lower_state = "VERTICAL"
            elif lower_angle > STATE_THRESHOLD_HORIZONTAL: lower_state = "HORIZONTAL"
            else: lower_state = "BENT"

        except Exception:
            pass

    # UI 출력 및 상태 종합 (기존과 동일)
    cv2.rectangle(image, (0, 0), (520, 120), (30, 30, 30), -1)
    u_color = (0, 255, 0) if upper_state == "VERTICAL" else (0, 0, 255)
    l_color = (0, 255, 0) if lower_state == "VERTICAL" else (0, 255, 255)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, f"UPPER(SHLD-HIP): {upper_state:8} | {upper_angle:4.1f}", (20, 45), font, 0.6, u_color, 1)
    cv2.putText(image, f"LOWER(HIP-KNEE) : {lower_state:8} | {lower_angle:4.1f}", (20, 90), font, 0.6, l_color, 1)

    full_status = "STABLE"
    if upper_state == "HORIZONTAL" and lower_state == "HORIZONTAL":
        full_status = "LYING DOWN"
    elif upper_state == "VERTICAL" and lower_state == "HORIZONTAL":
        full_status = "SITTING"

    cv2.putText(image, f"STATUS: {full_status}", (w-280, 65), font, 0.8, (255, 255, 255), 2)
    cv2.imshow('Body Part Monitor (Knee Ref)', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()