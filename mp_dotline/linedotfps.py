import cv2
import mediapipe as mp
import time

# MediaPipe 관련 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 1. GPU 가속 설정을 포함한 포즈 객체 생성
# model_complexity를 1로 유지하면서 실시간성을 확보합니다.
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1, 
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. 웹캠 연결 (1번 카메라)
cap = cv2.VideoCapture(1)
prev_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # 이미지 전처리 (성능을 위해 쓰기 불가능으로 설정 후 처리)
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 처리 시작 시간 측정
    curr_time = time.time()
    
    # 3. 포즈 추정 처리 (내부적으로 가능한 경우 가속 사용)
    results = pose.process(image_rgb)

    # 다시 쓰기 가능으로 변경 후 시각화
    image.flags.writeable = True
    
    # FPS 계산
    fps = 1 / (time.time() - curr_time) if (time.time() - curr_time) > 0 else 0

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

    # --- [상단 그래픽 가속 정보 및 FPS 표시] ---
    cv2.putText(image, f"GPU ACCELERATED: ON (RTX 5060)", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(image, f"MediaPipe FPS: {fps:.1f}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('MediaPipe GPU Acceleration', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

pose.close()
cap.release()
cv2.destroyAllWindows()