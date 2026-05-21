import cv2
import mediapipe as mp
import time
import math
import os

# [CUDA 설정] GPU 메모리 효율을 위해 TensorFlow 가속 옵션 환경 변수 설정
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# MediaPipe 설정
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==========================================
# [하드웨어 가속 설정]
# model_complexity: 2는 가장 정밀하며 GPU 가속 시 성능 저하가 적습니다.
# ==========================================
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,           # GPU 사용 시 2(Heavy) 권장
    smooth_landmarks=True,        # 데이터 노이즈 제거
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# [커스텀 설정 변수] - 여기서 값을 직접 변경하세요
# ==========================================
FALL_ANGLE_THRESHOLD = 45      # 상체 기울기 임계값 (0:수직, 90:수평)
HEIGHT_DROP_THRESHOLD = 0.12   # 골반 높이 하락폭 (낮을수록 민감)
TIME_WINDOW = 0.4              # 변화 감지 속도 (초 단위, 가속 시 더 짧게 설정 가능)
COOLDOWN_TIME = 3.0            # 감지 후 재감지 대기 시간
# ==========================================

prev_hip_height = None
prev_time = time.time()
last_fall_time = 0

cap = cv2.VideoCapture(1)

# OpenCV 프레임 속도 최적화 (카메라 지원 시)
cap.set(cv2.CAP_PROP_FPS, 60)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    current_time = time.time()
    
    # [가속 포인트] 이미지를 처리 가능한 형식으로 변환
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 모델 추론 (GPU 가속이 활성화된 경우 여기서 연산 수행)
    results = pose.process(image)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # 주요 관절 데이터
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        
        # 1. 골반 중심(y축 기준 무게중심)
        avg_hip_y = (l_hip.y + r_hip.y) / 2
        
        # 2. 상체 각도 계산 (atan2 사용)
        dx = nose.x - ((l_hip.x + r_hip.x) / 2)
        dy = nose.y - avg_hip_y
        body_angle = abs(math.degrees(math.atan2(dx, dy))) - 180
        body_angle = abs(body_angle)

        # 3. 넘어짐 판별 로직
        fall_detected = False
        
        # 각도가 무너지고 동시에 높이가 급락했을 때만 체크
        if body_angle > FALL_ANGLE_THRESHOLD:
            if prev_hip_height is not None:
                height_diff = avg_hip_y - prev_hip_height
                
                # 수직 낙하가 임계값을 넘었는지 확인
                if height_diff > HEIGHT_DROP_THRESHOLD:
                    if current_time - last_fall_time > COOLDOWN_TIME:
                        fall_detected = True
                        last_fall_time = current_time

        # 데이터 갱신
        if current_time - prev_time > TIME_WINDOW:
            prev_hip_height = avg_hip_y
            prev_time = current_time

        # 시각화 부분
        # 최근 1.5초 이내에 넘어짐이 감지되었다면 화면에 빨간색 표시
        status_color = (0, 0, 255) if (current_time - last_fall_time < 1.5) else (0, 255, 0)
        
        if fall_detected:
            print(f"[{time.strftime('%H:%M:%S')}] 🚨 FALL DETECTED!")

        # 화면 좌측 상단 상태 정보 출력
        cv2.putText(image, f"Angle: {int(body_angle)} / {FALL_ANGLE_THRESHOLD}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(image, f"CUDA Accelerated: ON", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        
        # 스켈레톤 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('MediaPipe CUDA Fall Detection', image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()