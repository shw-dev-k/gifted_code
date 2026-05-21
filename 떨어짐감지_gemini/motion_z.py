import cv2
import mediapipe as mp
import math

# [기준값 정의]
STATE_THRESHOLD_VERTICAL = 30 
STATE_THRESHOLD_HORIZONTAL = 60
# Z축 임계값: 무릎이 골반보다 이 수치만큼 카메라에 가까우면 앉은 것으로 간주
Z_THRESHOLD_SITTING = 0.15 

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# 웹캠 연결 (환경에 따라 0 또는 1)
cap = cv2.VideoCapture(0)

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
    z_depth_diff = 0.0

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            # 1. 상체 포인트 (어깨 중앙, 골반 중앙)
            shld_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
            shld_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            
            hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
            hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            hip_z = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].z + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].z) / 2

            # 2. 하체 포인트 (무릎 중앙)
            knee_x = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x + landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x) / 2
            knee_y = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y + landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y) / 2
            knee_z = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].z + landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].z) / 2

            # --- 상체 각도 계산 (수직 기준) ---
            u_dx = (shld_x - hip_x) * w
            u_dy = (shld_y - hip_y) * h
            upper_angle = abs(math.degrees(math.atan2(u_dx, u_dy)))
            if upper_angle > 90: upper_angle = abs(upper_angle - 180)

            # --- 하체 각도 계산 (Y축 기반) ---
            l_dx = (hip_x - knee_x) * w
            l_dy = (hip_y - knee_y) * h
            lower_angle = abs(math.degrees(math.atan2(l_dx, l_dy)))
            if lower_angle > 90: lower_angle = abs(lower_angle - 180)

            # --- 정면 앉음 보정 (Z축 깊이 기반) ---
            # 무릎의 z값이 골반의 z값보다 작을수록 카메라와 가까운 것
            z_depth_diff = hip_z - knee_z

            # --- 상태 판별 로직 ---
            # 상체 상태
            if upper_angle < STATE_THRESHOLD_VERTICAL: upper_state = "VERTICAL"
            elif upper_angle > STATE_THRESHOLD_HORIZONTAL: upper_state = "HORIZONTAL"
            else: upper_state = "SLANTED"

            # 하체 상태 (Y축 각도가 낮더라도 Z축 깊이 차이가 크면 앉은 것으로 간주)
            if lower_angle > STATE_THRESHOLD_HORIZONTAL or z_depth_diff > Z_THRESHOLD_SITTING:
                lower_state = "HORIZONTAL"
            elif lower_angle < STATE_THRESHOLD_VERTICAL:
                lower_state = "VERTICAL"
            else:
                lower_state = "BENT"

        except Exception:
            pass

    # --- UI 시각화 ---
    cv2.rectangle(image, (0, 0), (550, 140), (40, 40, 40), -1)
    
    u_color = (0, 255, 0) if upper_state == "VERTICAL" else (0, 0, 255)
    l_color = (0, 255, 0) if lower_state == "VERTICAL" else (0, 255, 255)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, f"UPPER (SHLD-HIP): {upper_state:8} | Ang: {upper_angle:4.1f}", (20, 40), font, 0.6, u_color, 1)
    cv2.putText(image, f"LOWER (HIP-KNEE) : {lower_state:8} | Ang: {lower_angle:4.1f}", (20, 80), font, 0.6, l_color, 1)
    cv2.putText(image, f"Z-DEPTH DIFF     : {z_depth_diff:8.3f}", (20, 120), font, 0.6, (255, 200, 0), 1)

    # 종합 상태 판별
    full_status = "STABLE"
    if upper_state == "HORIZONTAL" and lower_state == "HORIZONTAL":
        full_status = "LYING DOWN"
    elif upper_state == "VERTICAL" and lower_state == "HORIZONTAL":
        full_status = "SITTING"
    elif upper_state == "SLANTED" and lower_state == "BENT":
        full_status = "CROUCHED"

    # 상태 출력
    cv2.putText(image, f"STATUS: {full_status}", (w-320, 70), font, 0.9, (255, 255, 255), 2)

    cv2.imshow('Advanced Body Monitor', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()