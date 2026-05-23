import cv2
import mediapipe as mp
import math
import time

# [설정값]
STATE_THRESHOLD_HORIZONTAL = 65 
# 골반 높이 임계값 (0.0이 화면 맨 위, 1.0이 화면 맨 아래)
# 보통 서 있을 때 0.6~0.7, 바닥에 누우면 0.85~0.9 이상으로 커짐
HIP_LOW_THRESHOLD = 0.85  
LYING_TIME_THRESHOLD = 10.0  # 10초 지속 시 HELP

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

# --- 상태 관리 변수 ---
lying_start_time = 0
is_lying = False

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    now = time.time()
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    status_text = "NORMAL"

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        try:
            # 1. 주요 포인트 추출
            # 골반(Hip) 높이 체크 - 오인식 방지의 핵심
            hip_y = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
            
            # 상체 각도 계산 (어깨-골반)
            shld_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x) / 2
            shld_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y + landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y) / 2
            hip_x = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x + landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x) / 2
            # (hip_y는 위에서 계산함)

            u_dx, u_dy = (shld_x - hip_x) * w, (shld_y - hip_y) * h
            u_angle = abs(math.degrees(math.atan2(u_dx, u_dy)))
            if u_angle > 90: u_angle = abs(u_angle - 180)

            # 2. 누움 판정 조건 (복합 조건)
            # 조건: 상체 각도가 수평이면서 + 골반의 높이가 바닥(화면 하단)에 가까워야 함
            if u_angle > STATE_THRESHOLD_HORIZONTAL and hip_y > HIP_LOW_THRESHOLD:
                status_text = "LYING DOWN"
                if not is_lying:
                    is_lying = True
                    lying_start_time = now
            else:
                is_lying = False
                lying_start_time = 0
                status_text = "STABLE / BENDING"

        except: pass

    # --- 3. 알림 로직 (10초 지속 시) ---
    if is_lying:
        elapsed = now - lying_start_time
        if elapsed >= LYING_TIME_THRESHOLD:
            # 10초 경과 시 HELP 강제 출력
            if int(now * 4) % 2 == 0:
                cv2.rectangle(image, (0, 0), (w, h), (0, 0, 255), 30)
                cv2.putText(image, "HELP! HELP! HELP!", (w//2-380, h//2), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 10)
        else:
            # 10초 대기 중 메시지
            cv2.putText(image, f"CHECKING... {elapsed:.1f}s", (20, 110), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 255), 2)

    # 상단 정보 바
    cv2.rectangle(image, (0,0), (450, 80), (30,30,30), -1)
    cv2.putText(image, f"STATUS: {status_text}", (20, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Lying Detection (Height-based)', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()