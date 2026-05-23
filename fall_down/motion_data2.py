import cv2
import mediapipe as mp
import time
import math
import collections

# [기준값 정의]
FALL_ANGLE_THRESHOLD = 45      # 상체 기울기 임계값 (도)
HEIGHT_DROP_THRESHOLD = 0.12   # 골반 높이 하락 임계값
TIME_WINDOW = 0.4              

# MediaPipe Pose 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 데이터 부드럽게 처리를 위한 버퍼
hip_height_buffer = collections.deque(maxlen=10)
angle_buffer = collections.deque(maxlen=10)

# 웹캠 연결 (사용자 환경에 따라 0 또는 1)
cap = cv2.VideoCapture(1)

last_print_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("카메라를 찾을 수 없습니다.")
        break

    # 이미지 전처리
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    current_angle = 0.0
    current_height_drop = 0.0
    
    # UI 배경 (상단 검은색 바)
    overlay = image.copy()
    cv2.rectangle(overlay, (5, 5), (420, 120), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 스켈레톤 그리기
        mp_drawing.draw_landmarks(
            image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        try:
            # 1. 필요 좌표 추출 (어깨 및 골반)
            l_shld = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shld = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

            # 2. 중심점 계산
            # 어깨 중앙 (상체 상단)
            avg_shld_x = (l_shld.x + r_shld.x) / 2
            avg_shld_y = (l_shld.y + r_shld.y) / 2
            
            # 골반 중앙 (상체 하단 및 높이 기준)
            avg_hip_x = (l_hip.x + r_hip.x) / 2
            avg_hip_y = (l_hip.y + r_hip.y) / 2

            # 3. 기울기(Angle) 계산
            # 골반 중심을 기준으로 어깨 중심이 얼마나 기울었는지 계산
            dx = avg_shld_x - avg_hip_x
            dy = avg_shld_y - avg_hip_y
            
            # 수직선 대비 각도 산출
            raw_angle = math.degrees(math.atan2(dx, dy))
            current_angle = abs(abs(raw_angle) - 180)
            angle_buffer.append(current_angle)

            # 4. 높이 변화(Drop) 계산
            hip_height_buffer.append(avg_hip_y)
            if len(hip_height_buffer) > 1:
                # 버퍼의 첫 값(이전 위치)과 현재 값의 차이
                current_height_drop = hip_height_buffer[-1] - hip_height_buffer[0]
        
        except Exception as e:
            pass

    # --- 낙상 감지 및 터미널 알림 ---
    is_falling = current_angle > FALL_ANGLE_THRESHOLD or current_height_drop > HEIGHT_DROP_THRESHOLD
    
    if is_falling:
        if time.time() - last_print_time > 0.1:
            print(f"[WARNING] Fall Risk Detected!")
            print(f" > Angle: {current_angle:.1f}° | Drop: {current_height_drop:.3f}")
            print("-" * 40)
            last_print_time = time.time()

    # --- 화면 텍스트 출력 ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    alert_color = (0, 0, 255) # 위험 시 빨간색
    
    angle_color = alert_color if current_angle > FALL_ANGLE_THRESHOLD else white
    drop_color = alert_color if current_height_drop > HEIGHT_DROP_THRESHOLD else white

    cv2.putText(image, "--- BODY POSE MONITOR (SHOULDER) ---", (15, 30), font, 0.6, (255, 200, 0), 2)
    cv2.putText(image, f"SHOULDER ANGLE: {current_angle:.1f} deg", (20, 65), font, 0.6, angle_color, 2)
    cv2.putText(image, f"HIP DROP VALUE: {current_height_drop:.3f}", (20, 100), font, 0.6, drop_color, 2)

    # 상태 표시
    if is_falling:
        cv2.putText(image, "FALL DETECTED!", (20, 150), font, 1.2, alert_color, 3)

    cv2.imshow('Fall Detection System', image)

    # ESC 키로 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()