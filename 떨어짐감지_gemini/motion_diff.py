import cv2
import mediapipe as mp
import math

# [기준값 정의]
# 0도 = 완전 수직, 90도 = 완전 수평
STATE_THRESHOLD_VERTICAL = 30  # 30도 미만이면 서 있음
STATE_THRESHOLD_HORIZONTAL = 60 # 60도 이상이면 누움/앉음

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=1, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# 웹캠 연결 (0 또는 1)
cap = cv2.VideoCapture(1)

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
        
        # 가시성을 위해 스켈레톤 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            # 1. 상체 포인트 (어깨 중앙, 골반 중앙)
            shld_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
            shld_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
            hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2

            # 2. 하체 포인트 (골반 중앙, 발목 중앙)
            # 발목이 안 보일 경우를 대비해 무릎(KNEE)을 써도 되지만 일단 발목(ANKLE) 기준
            ank_x = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x + landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x) / 2
            ank_y = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y + landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y) / 2

            # --- 상체 각도 계산 (수직 기준) ---
            # 서 있을 때 dx=0, dy=값 이므로 atan2(dx, dy)는 0도가 나옴
            u_dx = (shld_x - hip_x) * w
            u_dy = (shld_y - hip_y) * h # 화면 좌표계 보정
            upper_angle = abs(math.degrees(math.atan2(u_dx, u_dy)))
            # 180도 근처로 가면(똑바로 선 경우) 0도로 변환
            if upper_angle > 90:
                upper_angle = abs(upper_angle - 180)

            # --- 하체 각도 계산 (수직 기준) ---
            l_dx = (hip_x - ank_x) * w
            l_dy = (hip_y - ank_y) * h
            lower_angle = abs(math.degrees(math.atan2(l_dx, l_dy)))
            if lower_angle > 90:
                lower_angle = abs(lower_angle - 180)

            # --- 개별 상태 판별 ---
            # 상체
            if upper_angle < STATE_THRESHOLD_VERTICAL: upper_state = "VERTICAL"
            elif upper_angle > STATE_THRESHOLD_HORIZONTAL: upper_state = "HORIZONTAL"
            else: upper_state = "SLANTED"

            # 하체
            if lower_angle < STATE_THRESHOLD_VERTICAL: lower_state = "VERTICAL"
            elif lower_angle > STATE_THRESHOLD_HORIZONTAL: lower_state = "HORIZONTAL"
            else: lower_state = "BENT"

        except Exception:
            pass

    # --- UI 시각화 ---
    # 배경 바
    cv2.rectangle(image, (0, 0), (500, 120), (30, 30, 30), -1)
    
    # 텍스트 출력 컬러 설정
    u_color = (0, 255, 0) if upper_state == "VERTICAL" else (0, 0, 255)
    l_color = (0, 255, 0) if lower_state == "VERTICAL" else (0, 255, 255)

    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, f"UPPER: {upper_state:10} | Ang: {upper_angle:4.1f}", (20, 45), font, 0.7, u_color, 1)
    cv2.putText(image, f"LOWER: {lower_state:10} | Ang: {lower_angle:4.1f}", (20, 90), font, 0.7, l_color, 1)

    # 전체적인 판례 (종합 상태)
    full_status = "STABLE"
    if upper_state == "HORIZONTAL" and lower_state == "HORIZONTAL":
        full_status = "LYING DOWN (FALL)"
    elif upper_state == "VERTICAL" and lower_state == "HORIZONTAL":
        full_status = "SITTING"

    cv2.putText(image, f"STATUS: {full_status}", (w-300, 65), font, 0.8, (255, 255, 255), 2)

    cv2.imshow('Separate Body Part Monitor', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()