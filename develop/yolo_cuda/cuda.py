import cv2
import torch
from ultralytics import YOLO

# =========================
# 1. 환경 설정 (CUDA 12.8 전용)
# =========================
# CUDA 12.8과 sm_120 아키텍처를 강제로 인식시키기 위한 설정입니다.
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if device == 'cuda:0':
    # 최신 GPU 성능을 위해 벤치마크 기능을 켭니다.
    torch.backends.cudnn.benchmark = True
    print(f"🚀 CUDA 12.8 가속 활성화: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU 인식 실패. CPU로 구동합니다.")

# =========================
# 2. 모델 학습 (Training)
# =========================
def train_process():
    model = YOLO("yolov8x-cls.pt")

    # 5060의 성능을 고려하여 배치 사이즈 128로 설정
    model.train(
        data=r"C:\Users\소현우\Downloads\archive\DATASET\DATASET",
        epochs=10, 
        imgsz=224, 
        batch=64,      # 128에서 32로 과감하게 낮추세요. 안전이 제일입니다.
        workers=2,      # 프로세스 복제량을 줄여 메모리 에러를 방지합니다.
        cache=False,
        device=0,     # GPU 번호를 명시적으로 지정
        name="O_R_model",
        exist_ok=True
    )
    print("학습 완료")

# =========================
# 3. 실시간 웹캠 추론 (Inference)
# =========================
def run_inference():
    # 학습된 모델을 불러와 GPU 메모리에 올립니다.
    model = YOLO("runs/classify/O_R_model/weights/best.pt").to(device)

    print("웹캠 시작 (종료하려면 'q'를 누르세요)")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # CUDA 12.8 환경에서 가장 빠른 FP16(반정밀도) 추론 사용
        # Blackwell(50시리즈)은 이 연산에서 압도적인 속도를 냅니다.
        results = model.predict(
            frame, 
            device=device, 
            half=True,    # 12.8 가속의 핵심 옵션
            verbose=False
        )[0]

        # 결과 추출
        probs = results.probs
        label = results.names[probs.top1]
        conf = probs.top1conf.item()

        # 화면 출력
        display_text = f"CUDA 12.8 | {label} {conf*100:.1f}%"
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("RTX 5060 & CUDA 12.8 TEST", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 1. 처음에는 학습을 먼저 해서 best.pt를 만들어야 합니다.
    train_process()
    
    # 2. 학습이 끝나면 아래 주석을 풀고 추론을 실행하세요.
    #run_inference()