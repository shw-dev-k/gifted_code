import cv2
import mediapipe as mp
import time
import math
import collections

# [기준값 정의]
FALL_ANGLE_THRESHOLD = 45      
HEIGHT_DROP_THRESHOLD = 0.12   
TIME_WINDOW = 0.4              

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(model_complexity=1)

hip_height_buffer = collections.deque(maxlen=10)
angle_buffer = collections.deque(maxlen=10)

cap = cv2.VideoCapture(1)

# 터미널 출력 간격 조절을 위한 변수 (선택 사항)
last_print_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    current_angle = 0.0
    current_height_drop = 0.0
    
    overlay = image.copy()
    cv2.rectangle(overlay, (5, 5), (380, 120), (0, 0, 0), -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            
            avg_hip_y = (l_hip.y + r_hip.y) / 2
            hip_height_buffer.append(avg_hip_y)
            
            dx = nose.x - ((l_hip.x + r_hip.x) / 2)
            dy = nose.y - avg_hip_y
            current_angle = abs(math.degrees(math.atan2(dx, dy))) - 180
            current_angle = abs(current_angle)
            angle_buffer.append(current_angle)

            if len(hip_height_buffer) > 1:
                current_height_drop = hip_height_buffer[-1] - hip_height_buffer[0]
        
        except Exception as e:
            pass

    # --- 터미널 출력 로직 추가 ---
    # 각도나 낙하폭 중 하나라도 임계값을 넘으면 터미널에 경고 메시지 출력
    if current_angle > FALL_ANGLE_THRESHOLD or current_height_drop > HEIGHT_DROP_THRESHOLD:
        # 너무 빠르게 출력되는 것을 방지하기 위해 0.1초마다 출력 (선택 사항)
        if time.time() - last_print_time > 0.1:
            print(f"[WARNING] Critical Value Detected!")
            print(f" > Angle: {current_angle:.1f} deg | Drop: {current_height_drop:.3f}")
            print("-" * 40)
            last_print_time = time.time()
    # -----------------------

    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    
    angle_color = (0, 255, 255) if current_angle > FALL_ANGLE_THRESHOLD else white
    drop_color = (0, 255, 255) if current_height_drop > HEIGHT_DROP_THRESHOLD else white

    cv2.putText(image, "--- REAL-TIME BODY MONITOR ---", (15, 30), font, 0.6, (255, 200, 0), 2)
    cv2.putText(image, f"LIVE ANGLE: {current_angle:.1f} deg", (20, 65), font, 0.6, angle_color, 2)
    cv2.putText(image, f"LIVE DROP : {current_height_drop:.3f}", (20, 100), font, 0.6, drop_color, 2)

    cv2.imshow('Live Parameter Monitor', image)

    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()