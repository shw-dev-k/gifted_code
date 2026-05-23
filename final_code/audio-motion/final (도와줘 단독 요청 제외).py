import cv2
import mediapipe as mp
import numpy as np
import time
import sys
from collections import deque
from google import genai
from PIL import Image
import threading
import speech_recognition as sr

# 음성 라이브러리 경로 추가
sys.path.insert(0, r"C:\Users\소현우\Desktop\project\ai_def\sound")

from speech import speak
from listener import listen_text
from detector import is_danger, is_safe

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 1. Gemini 클라이언트 설정 (사용자 API 키 반영)
client = genai.Client(api_key="your_api_key_here")

def get_body_state(keypoints):
    try:
        ls = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        rs = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        lh = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
        rh = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # 누워 있을 때는 선이 흐려지므로 가시성 기준을 0.2로 완화
        if any(kp.visibility < 0.2 for kp in [ls, rs, lh, rh]):
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

# 최근 움직임 값들을 저장해 평균을 내기 위한 큐 (프레임 튐 방지)
movement_history = deque(maxlen=5)

def check_immobility(current_landmarks, prev_landmarks, threshold=0.025):
    if prev_landmarks is None or current_landmarks is None:
        return False
        
    # 누워 있을 때 그나마 덜 흔들리는 몸통과 주요 관절 위주로 체크
    check_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.RIGHT_KNEE.value
    ]
    
    total_movement = 0
    valid_points = 0
    
    for idx in check_indices:
        kp_curr = current_landmarks.landmark[idx]
        kp_prev = prev_landmarks.landmark[idx]
        
        # 선이 흐릿하게 찍혀도 일단 계산에 포함하도록 가시성 기준을 0.2로 낮춤
        if kp_curr.visibility < 0.2 or kp_prev.visibility < 0.2:
            continue
            
        dist = np.sqrt((kp_curr.x - kp_prev.x)**2 + (kp_curr.y - kp_prev.y)**2)
        total_movement += dist
        valid_points += 1
        
    if valid_points == 0:
        # 관절이 아예 안 보일 정도로 가려진 경우, 움직임이 없다고 가정(True)하거나 
        # 이전 상태를 유지하는 것이 안전합니다. 여기서는 일단 False 처리 후 큐로 보완합니다.
        return False
        
    avg_movement = total_movement / valid_points
    movement_history.append(avg_movement)
    
    # 최근 5프레임의 평균 움직임 계산
    smoothed_movement = sum(movement_history) / len(movement_history)
    
    # 평균 움직임이 임계값보다 작으면 정지 상태로 판정
    return smoothed_movement < threshold

# Gemini API 분석을 백그라운드에서 실행할 스레드 함수
def request_gemini_analysis(frame_to_analyze):
    print("\n[시스템] 낙상 및 정지 상태 감지! Gemini 분석을 시작합니다...")
    try:
        rgb_frame = cv2.cvtColor(frame_to_analyze, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        prompt = "이 이미지에 사람이 쓰러져 있는 상황인가요? 위험 상태를 진단하고 환자의 자세와 주변 환경을 한국어로 간결하고 명확하게 설명해 주세요."

        print(f"[프로젝트 프롬프트]: {prompt}")
        print("[Gemini]: 이미지를 분석 중입니다...")

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[pil_img, prompt]
        )

        gemini_text = response.text
        print(f"\n[Gemini Response]:\n{gemini_text}\n")
        print("-" * 50)

        # Gemini 분석 결과 음성 안내 (앞 60자만 읽어 너무 길지 않게)
        speak("낙상이 감지되었습니다. " + gemini_text[:60])

        # 사용자 확인 인터랙션 (최대 2회 시도)
        emergency = True

        for attempt in range(2):
            speak("괜찮으세요?")

            try:
                response_text, _ = listen_text(timeout=10)
                print("사용자 응답:", response_text)

                if is_safe(response_text):
                    emergency = False
                    break
                elif is_danger(response_text):
                    emergency = True
                    break
                else:
                    if attempt == 0:
                        speak("다시 한 번 말씀해 주세요.")

            except sr.WaitTimeoutError:
                print("응답 시간 초과")
                if attempt == 0:
                    speak("다시 한 번 말씀해 주세요.")

            except Exception as e:
                print("응답 인식 실패:", e)
                if attempt == 0:
                    speak("다시 한 번 말씀해 주세요.")

        if emergency:
            print("\n" + "=" * 50)
            print("!!! HELP !!! 긴급 상황 감지 - 즉시 도움 요청")
            print("=" * 50 + "\n")
            speak("전화로 도움을 요청하겠습니다.")
        else:
            speak("다행이군요")

    except Exception as e:
        print(f"\n[Gemini 에러 발생]: {e}\n")

# 카메라 설정 (1번 포트가 안 열리면 0으로 변경하세요)
cap = cv2.VideoCapture(0)

# 해상도 자동 설정 유도
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

ret, frame = cap.read()
if not ret:
    print("카메라를 읽을 수 없습니다.")
    exit()

h, w = frame.shape[:2]
print(f"[안내] 카메라 화질이 자동으로 설정되었습니다: {w} x {h}")

cv2.namedWindow('Fall Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Fall Detection', w, h)

HISTORY_SEC = 1.0
FALL_SPEED  = 35.0
FALL_ANGLE  = 40.0
HELP_SEC    = 5.0

# [수정] 정지 상태 판단 임계값 대폭 상향 (0.008 -> 0.025)
# 관절 선이 미세하게 요동쳐도 이 값보다 작으면 '정지'로 인식합니다.
IMMOBILE_THRES = 0.025

angle_history = deque()
prev_pose_landmarks = None

fall_confirmed  = False
fall_duration   = None
immobile_since  = None
help_triggered  = False
gemini_analyzed = False  

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
                        gemini_analyzed = False  
                        angle_history.clear()
                        movement_history.clear()

                # 낙상 확정 상태에서 '움직임 없음' 판단 로직
                if fall_confirmed:
                    is_immobile = check_immobility(results.pose_landmarks, prev_pose_landmarks, threshold=IMMOBILE_THRES)

                    if is_immobile:
                        if immobile_since is None:
                            immobile_since = current_time
                        immobile_duration = current_time - immobile_since
                        if immobile_duration >= HELP_SEC:
                            help_triggered = True
                            
                            if not gemini_analyzed:
                                gemini_analyzed = True
                                t = threading.Thread(target=request_gemini_analysis, args=(image.copy(),))
                                t.start()
                    else:
                        # 순간적인 튐으로 인해 타이머가 초기화되는 것을 방지하기 위해 
                        # 완전히 움직임이 감지되었을 때만 타이머 리셋
                        immobile_since = current_time

                prev_pose_landmarks = results.pose_landmarks

                # 화면 UI 크기 조정
                scale_factor = max(0.5, w / 1920.0)
                
                if help_triggered:
                    if int(current_time * 2) % 2 == 0:
                        cv2.putText(image, '!! HELP !!', (int(50*scale_factor), int(80*scale_factor)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3.0 * scale_factor, (0, 0, 255), max(1, int(7*scale_factor)))

                elif fall_confirmed:
                    cv2.putText(image, 'FALL DETECTED', (int(50*scale_factor), int(80*scale_factor)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.5 * scale_factor, (0, 0, 255), max(1, int(5*scale_factor)))
                    if fall_duration is not None:
                        cv2.putText(image, f'Fall Time: {fall_duration:.2f}s', (int(50*scale_factor), int(160*scale_factor)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0 * scale_factor, (0, 165, 255), max(1, int(4*scale_factor)))
                    if immobile_since is not None:
                        immobile_duration = current_time - immobile_since
                        remaining = HELP_SEC - immobile_duration
                        cv2.putText(image, f'HELP until: {max(0.0, remaining):.1f}s', (int(50*scale_factor), int(240*scale_factor)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.8 * scale_factor, (0, 255, 255), max(1, int(4*scale_factor)))

                cv2.putText(image,
                            f'spine:{spine_angle:.1f}  ratio:{aspect_ratio:.2f}  speed:{angle_change_rate:.1f}',
                            (int(20*scale_factor), image.shape[0] - int(20*scale_factor)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale_factor, (200, 200, 200), max(1, int(2*scale_factor)))

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
            # 사람이 쓰러져서 선이 아예 증발해 버린 경우에도 
            # 기존에 이미 낙상이 감지된 상태였다면 정지(Immobile) 상태가 이어지고 있는 것으로 처리함
            if fall_confirmed and immobile_since is not None:
                immobile_duration = current_time - immobile_since
                if immobile_duration >= HELP_SEC and not gemini_analyzed:
                    help_triggered = True
                    gemini_analyzed = True
                    t = threading.Thread(target=request_gemini_analysis, args=(image.copy(),))
                    t.start()
                    
            prev_pose_landmarks = None

        cv2.imshow('Fall Detection', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()