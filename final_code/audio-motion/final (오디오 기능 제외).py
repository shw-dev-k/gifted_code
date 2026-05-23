import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from google import genai
from PIL import Image
import threading

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 1. Gemini 클라이언트 설정 (사용자 API 키 반영)
client = genai.Client(api_key="YOUR_GEMINI_API_KEY")

def get_body_state(keypoints):
    try:
        ls = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]

        if any(kp.visibility < 0.4 for kp in [ls, rs, lh, rh]):
            return None, None

        mid_shoulder = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
        mid_hip      = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])

        spine_vec   = mid_hip - mid_shoulder
        spine_angle = abs(np.degrees(np.arctan2(abs(spine_vec[1]),
                                                abs(spine_vec[0]) + 1e-6)))

        all_x = [keypoints[i].x for i in range(33)]
        all_y = [keypoints[i].y for i in range(33)]
        aspect_ratio = (max(all_x) - min(all_x)) / (max(all_y) - min(all_y) + 1e-6)

        return spine_angle, aspect_ratio
    except:
        return None, None

def check_immobility(current_landmarks, prev_landmarks, threshold=0.015):
    if prev_landmarks is None or current_landmarks is None:
        return False
        
    check_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_WRIST.value, mp_pose.PoseLandmark.RIGHT_WRIST.value
    ]
    
    total_movement = 0
    valid_points = 0
    
    for idx in check_indices:
        kp_curr = current_landmarks.landmark[idx]
        kp_prev = prev_landmarks.landmark[idx]
        
        if kp_curr.visibility < 0.4 or kp_prev.visibility < 0.4:
            continue
            
        dist = np.sqrt((kp_curr.x - kp_prev.x)**2 + (kp_curr.y - kp_prev.y)**2)
        total_movement += dist
        valid_points += 1
        
    if valid_points == 0:
        return False
        
    avg_movement = total_movement / valid_points
    return avg_movement < threshold

# Gemini API 분석을 백그라운드에서 실행할 스레드 함수
def request_gemini_analysis(frame_to_analyze):
    print("\n[시스템] 낙상 및 정지 상태 감지! Gemini 분석을 시작합니다...")
    try:
        # OpenCV BGR 영상을 RGB로 변환 후 PIL 이미지로 변환
        rgb_frame = cv2.cvtColor(frame_to_analyze, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        prompt = "이 이미지에 사람이 쓰러져 있는 상황인가요? 위험 상태를 진단하고 환자의 자세와 주변 환경을 한국어로 간결하고 명확하게 설명해 주세요."
        
        print(f"[프로젝트 프롬프트]: {prompt}")
        print("[Gemini]: 이미지를 분석 중입니다...")

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[pil_img, prompt]
        )
        print(f"\n[Gemini Response]:\n{response.text}\n")
        print("-" * 50)
    except Exception as e:
        print(f"\n[Gemini 에러 발생]: {e}\n")

# 카메라 설정
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

ret, frame = cap.read()
if not ret:
    print("카메라를 읽을 수 없습니다.")
    exit()

h, w = frame.shape[:2]
cv2.namedWindow('Fall Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fall Detection', w, h)

HISTORY_SEC = 1.0
FALL_SPEED  = 35.0
FALL_ANGLE  = 40.0
HELP_SEC    = 5.0
IMMOBILE_THRES = 0.008

angle_history = deque()
prev_pose_landmarks = None

fall_confirmed  = False
fall_duration   = None
immobile_since  = None
help_triggered  = False
gemini_analyzed = False  # [추가] 연속 호출을 방지하기 위한 플래그

flashing        = False
last_flash_time = 0
flash_interval  = 0.5
mask_visible    = False

print("=" * 50)
print("실시간 낙상 감지 & Gemini 비전 연동 프로그램이 시작되었습니다.")
print("- 낙상 후 5초간 정지 시 자동 위험 분석 요청")
print("- [q 키]: 프로그램 종료")
print("=" * 50)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image     = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = pose.process(image_rgb)

        current_time = time.time()

        if results.pose_landmarks:
            keypoints = results.pose_landmarks.landmark
            spine_angle, aspect_ratio = get_body_state(keypoints)

            if spine_angle is not None:
                angle_history.append((current_time, spine_angle))
                while angle_history and current_time - angle_history[0][0] > HISTORY_SEC:
                    angle_history.popleft()

                is_lying = (spine_angle < FALL_ANGLE or aspect_ratio > 1.3)

                angle_change_rate = 0.0
                if len(angle_history) >= 2:
                    oldest_t, oldest_a = angle_history[0]
                    dt = current_time - oldest_t
                    if dt > 0:
                        angle_change_rate = (oldest_a - spine_angle) / dt

                is_fall = is_lying and angle_change_rate > FALL_SPEED

                if is_fall:
                    if not fall_confirmed:
                        fall_confirmed = True
                        fall_duration  = current_time - angle_history[0][0]
                        immobile_since = current_time
                    flashing = True
                else:
                    if not is_lying:
                        fall_confirmed  = False
                        fall_duration   = None
                        flashing        = False
                        mask_visible    = False
                        immobile_since  = None
                        help_triggered  = False
                        gemini_analyzed = False  # 사람이 일어나면 플래그 리셋
                        angle_history.clear()

                # 낙상 확정 상태에서 '움직임 없음' 판단 로직
                if fall_confirmed:
                    is_immobile = check_immobility(results.pose_landmarks, prev_pose_landmarks, threshold=IMMOBILE_THRES)

                    if is_immobile:
                        if immobile_since is None:
                            immobile_since = current_time
                        immobile_duration = current_time - immobile_since
                        if immobile_duration >= HELP_SEC:
                            help_triggered = True
                            
                            # [추가] help가 트리거되었고 아직 Gemini 분석을 보내지 않았다면 백그라운드 호출
                            if not gemini_analyzed:
                                gemini_analyzed = True
                                # 영상이 멈추지 않게 스레드로 이미지 분석 전달
                                t = threading.Thread(target=request_gemini_analysis, args=(image.copy(),))
                                t.start()
                    else:
                        immobile_since = current_time

                prev_pose_landmarks = results.pose_landmarks

                # ── 화면 출력 ───────────────────────────────
                if help_triggered:
                    if int(current_time * 2) % 2 == 0:
                        cv2.putText(image, '!! HELP !!', (50, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 7)

                elif fall_confirmed:
                    cv2.putText(image, 'FALL DETECTED', (50, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
                    if fall_duration is not None:
                        cv2.putText(image, f'Fall Time: {fall_duration:.2f}s', (50, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 165, 255), 4)
                    if immobile_since is not None:
                        immobile_duration = current_time - immobile_since
                        remaining = HELP_SEC - immobile_duration
                        cv2.putText(image, f'HELP until: {max(0.0, remaining):.1f}s', (50, 240),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)

                cv2.putText(image,
                            f'spine:{spine_angle:.1f}  ratio:{aspect_ratio:.2f}  speed:{angle_change_rate:.1f}',
                            (20, image.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)

            if flashing and not help_triggered:
                if current_time - last_flash_time >= flash_interval:
                    mask_visible    = not mask_visible
                    last_flash_time = current_time
                if mask_visible:
                    overlay = image.copy()
                    cv2.rectangle(overlay, (0, 0),
                                  (image.shape[1], image.shape[0]), (0, 0, 255), -1)
                    image = cv2.addWeighted(overlay, 0.4, image, 0.6, 0)

            mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                      mp_pose.POSE_CONNECTIONS)
        else:
            prev_pose_landmarks = None

        cv2.imshow('Fall Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()