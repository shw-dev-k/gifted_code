import cv2
import mediapipe as mp
import numpy as np
import time
import math

# --- 초기화 및 설정 ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

STATE_THRESHOLD_VERTICAL = 30
STATE_THRESHOLD_HORIZONTAL = 60

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
        return not (angle < 30 or angle > 150)
    except:
        return False

# 카메라 시작 (0번 기본 카메라 사용)
cap = cv2.VideoCapture(1)

# --- STEP 1: 낙하 시간 측정 모드 ---
fall_valid = False
total_fall_duration = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    fall_detected = False
    fall_start_time = None

    print("상태: 낙하 시간 측정 중...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

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
                    total_fall_duration = current_time - fall_start_time
                    cv2.putText(image, f'Fall Process: {total_fall_duration:.2f}s', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                    if total_fall_duration >= 0.2:
                        cv2.putText(image, 'FALL DETECTED!', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                # 떨어지는 동작이 멈췄을 때(조건 해제) 3초 이내 판단
                if fall_detected:
                    if total_fall_duration <= 3.0:
                        fall_valid = True
                        print(f"판단: 3초 이내 낙하 확인 ({total_fall_duration:.2f}s). 모니터링 전환.")
                        break
                    else:
                        print(f"판단: 3초 초과 ({total_fall_duration:.2f}s). 초기화.")
                        fall_detected = False
                        fall_start_time = None
                        total_fall_duration = 0

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Phase 1: Fall Timer', image)
        if cv2.waitKey(10) & 0xFF == ord('q'): break

# --- STEP 2: 자세 유지 감시 모드 (3초 이내 낙하 시 실행) ---
if fall_valid:
    lying_start_time = None
    
    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success: break

            image = cv2.flip(image, 1)
            h, w, _ = image.shape
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            upper_state, lower_state, full_status = "IDLE", "IDLE", "STABLE"
            upper_angle, lower_angle = 0.0, 0.0

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                try:
                    shld_x = (landmarks[11].x + landmarks[12].x) / 2
                    shld_y = (landmarks[11].y + landmarks[12].y) / 2
                    hip_x = (landmarks[23].x + landmarks[24].x) / 2
                    hip_y = (landmarks[23].y + landmarks[24].y) / 2
                    knee_x = (landmarks[25].x + landmarks[26].x) / 2
                    knee_y = (landmarks[25].y + landmarks[26].y) / 2

                    # 상체 각도
                    u_dx, u_dy = (shld_x - hip_x) * w, (shld_y - hip_y) * h
                    upper_angle = abs(math.degrees(math.atan2(u_dx, u_dy)))
                    if upper_angle > 90: upper_angle = abs(upper_angle - 180)

                    # 하체 각도
                    l_dx, l_dy = (hip_x - knee_x) * w, (hip_y - knee_y) * h
                    lower_angle = abs(math.degrees(math.atan2(l_dx, l_dy)))
                    if lower_angle > 90: lower_angle = abs(lower_angle - 180)

                    if upper_angle < STATE_THRESHOLD_VERTICAL: upper_state = "VERTICAL"
                    elif upper_angle > STATE_THRESHOLD_HORIZONTAL: upper_state = "HORIZONTAL"
                    else: upper_state = "SLANTED"

                    if lower_angle < STATE_THRESHOLD_VERTICAL: lower_state = "VERTICAL"
                    elif lower_angle > STATE_THRESHOLD_HORIZONTAL: lower_state = "HORIZONTAL"
                    else: lower_state = "BENT"

                    if upper_state == "HORIZONTAL" and lower_state == "HORIZONTAL":
                        full_status = "LYING DOWN"
                    elif upper_state == "VERTICAL" and lower_state == "HORIZONTAL":
                        full_status = "SITTING"
                except: pass

            # 10초 지속 측정
            if full_status == "LYING DOWN":
                if lying_start_time is None:
                    lying_start_time = time.time()
                
                duration = time.time() - lying_start_time
                cv2.putText(image, f"Lying: {duration:.1f}s", (w-250, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if duration >= 10.0:
                    cv2.putText(image, "HELP!", (w//2-100, h//2), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0, 0, 255), 5)
            else:
                lying_start_time = None

            # UI 레이아웃
            cv2.rectangle(image, (0, 0), (520, 120), (30, 30, 30), -1)
            cv2.putText(image, f"UPPER: {upper_state} | {upper_angle:.1f}", (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, f"LOWER: {lower_state} | {lower_angle:.1f}", (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(image, f"STATUS: {full_status}", (w-280, 65), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Phase 2: Lying Monitor', image)
            if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

#https://42morrow.tistory.com/entry/%EC%93%B0%EB%9F%AC%EC%A7%90-%EA%B0%90%EC%A7%80-Fall-Detection
#https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos?tab=readme-ov-file