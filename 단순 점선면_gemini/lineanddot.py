import cv2
import mediapipe as mp

# 1. MediaPipe Pose 솔루션 초기화
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # 점과 선을 그리는 도구
mp_drawing_styles = mp.solutions.drawing_styles  # 예쁜 스타일 적용

# 포즈 탐지 객체 생성
# model_complexity=1은 성능과 정확도의 균형이 가장 좋습니다.
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. 웹캠 연결
cap = cv2.VideoCapture(1)  # 0번 카메라는 보통 기본 웹캠입니다.

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 불러올 수 없습니다.")
        break

    # 성능 향상을 위해 이미지를 RGB로 변환 (MediaPipe는 RGB를 사용)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 3. 포즈 추정 처리
    results = pose.process(image_rgb)

    # 4. 결과 시각화 (원본 BGR 이미지 위에 그리기)
    if results.pose_landmarks:
        # LANDMARKS (점)와 CONNECTIONS (선)를 그립니다.
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            # landmark_drawing_spec: 점 스타일 (빨간색)
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            # connection_drawing_spec: 선 스타일 (초록색)
            # connection_drawing_spec=mp_drawing_styles.get_default_pose_connections_style()
        )

    # 5. 화면 표시
    cv2.imshow('MediaPipe Pose Skeleton', image)
    
    # 'ESC' 키를 누르면 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 6. 자원 해제
pose.close()
cap.release()
cv2.destroyAllWindows()
