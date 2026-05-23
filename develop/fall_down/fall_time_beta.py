import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe 및 도구 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(cosine_angle))

def is_fall_condition_met(keypoints):
    try:
        ls, rs = keypoints[11], keypoints[12] # Shoulders
        lh, rh = keypoints[23], keypoints[24] # Hips
        lk, rk = keypoints[25], keypoints[26] # Knees

        if ls.visibility < 0.5 or rs.visibility < 0.5 or lh.visibility < 0.5:
            return False

        shoulder = [(ls.x + rs.x) / 2, (ls.y + rs.y) / 2]
        hip = [(lh.x + rh.x) / 2, (lh.y + rh.y) / 2]
        knee = [(lk.x + rk.x) / 2, (lk.y + rk.y) / 2]

        angle = calculate_angle(shoulder, hip, knee)
        # 서 있는 상태가 아닌(굽혀진) 각도일 때 True
        return not (angle < 30 or angle > 150)
    except:
        return False

# 비디오 로드 및 창 설정
cap = cv2.VideoCapture('20260509_161301.mp4')
cv2.namedWindow('Fall Detection', cv2.WINDOW_NORMAL)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    fall_detected = False
    fall_start_time = None
    total_fall_duration = 0  # 쓰러지는 데 걸린 시간 저장 변수

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 영상 리사이징 (화면 맞춤)
        frame = cv2.resize(frame, (800, int(800 * frame.shape[0] / frame.shape[1])))
        image = cv2.flip(frame, 1)
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            fall_condition = is_fall_condition_met(results.pose_landmarks.landmark)
            current_time = time.time()

            if fall_condition:
                if not fall_detected:
                    fall_start_time = current_time
                    fall_detected = True
                else:
                    # 현재 진행 중인 쓰러짐 시간 계산
                    total_fall_duration = current_time - fall_start_time
                    
                    # 화면에 소요 시간 표시 (소수점 둘째자리까지)
                    cv2.putText(image, f'Fall Process: {total_fall_duration:.2f}s', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                    if total_fall_duration >= 0.2: # 0.2초 이상 유지되면 쓰러짐 확정
                        cv2.putText(image, 'FALL DETECTED!', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                fall_detected = False
                fall_start_time = None
                total_fall_duration = 0

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Fall Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()