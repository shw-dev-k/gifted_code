import cv2
import numpy as np
import torch
import time  # 시간 측정을 위해 추가
from ultralytics import YOLO

# 1. 모델 설정 및 장치 할당
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8x-pose.pt').to(device)

print("-" * 30)
print(f"실행 장치: {device}")
if device == 'cuda':
    print(f"GPU 모델: {torch.cuda.get_device_name(0)}")
print("-" * 30)

# 2. 물리 및 필터 하이퍼파라미터
DT = 1.0           
DAMPING = 0.8      
GRAVITY = 9.8      
ALPHA = 0.3        

# 3. 가변 바닥 및 상태 변수
current_floor_y = 480
stay_count = 0
candidate_y = 0
prev_kpts = None
smooth_v = np.zeros((17, 2))

# FPS 계산을 위한 변수 초기화
prev_time = 0
curr_time = 0

# 4. 스켈레톤 연결 쌍
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
    (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # --- [FPS 계산 루직] ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time

    # YOLOv8x 추론
    results = model(frame, verbose=False, device=device)

    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        curr_kpts = results[0].keypoints.data[0].cpu().numpy()

        if prev_kpts is not None:
            # [STEP 1: 지지점 파악]
            if curr_kpts[15][2] > 0.5 and curr_kpts[16][2] > 0.5:
                foot_base_x = (curr_kpts[15][0] + curr_kpts[16][0]) / 2
            else:
                foot_base_x = (curr_kpts[11][0] + curr_kpts[12][0]) / 2
                cv2.putText(frame, "VIRTUAL FOOT MODE", (20, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # [STEP 2: 가변 바닥 업데이트]
            pelvis_y = (curr_kpts[11][1] + curr_kpts[12][1]) / 2
            if abs(pelvis_y - candidate_y) < 10:
                stay_count += 1
            else:
                candidate_y = pelvis_y
                stay_count = 0

            if stay_count > 30:
                current_floor_y = candidate_y + 100
                stay_count = 0

            # [STEP 3: 유령 물리 엔진]
            future_kpts = []
            for i in range(17):
                raw_v = curr_kpts[i][:2] - prev_kpts[i][:2]
                smooth_v[i] = ALPHA * raw_v + (1 - ALPHA) * smooth_v[i]

                vx, vy = smooth_v[i] * DAMPING
                f_x = curr_kpts[i][0] + (vx * DT)
                f_y = curr_kpts[i][1] + (vy * DT) + (0.5 * GRAVITY * (DT ** 2))

                if f_y > current_floor_y:
                    f_y = current_floor_y
                future_kpts.append([f_x, f_y])

            # [STEP 4: 시각화 및 낙상 경고]
            cv2.line(frame, (0, int(current_floor_y)), (int(frame.shape[1]), int(current_floor_y)), (255, 255, 0), 2)

            for p1_idx, p2_idx in SKELETON:
                p1 = (int(future_kpts[p1_idx][0]), int(future_kpts[p1_idx][1]))
                p2 = (int(future_kpts[p2_idx][0]), int(future_kpts[p2_idx][1]))
                cv2.line(frame, p1, p2, (200, 200, 200), 2)

            for kpt in future_kpts:
                cv2.circle(frame, (int(kpt[0]), int(kpt[1])), 3, (255, 0, 0), -1)

            f_pelvis_x = (future_kpts[11][0] + future_kpts[12][0]) / 2
            if abs(f_pelvis_x - foot_base_x) > 120:
                cv2.putText(frame, "FALL PREDICTED!", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        prev_kpts = curr_kpts

    # --- [최상단 FPS 표시] ---
    # 가독성을 위해 배경 사각형을 살짝 그려줄 수도 있지만, 요청하신 대로 텍스트만 깔끔하게 넣었습니다.
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow('YOLOv8x CUDA Fall Simulation', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()