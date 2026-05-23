import cv2
import torch
from ultralytics import YOLO

# ============================================================
# [ 사용자 설정 섹션 ] - 이 부분의 값만 수정하여 제어하세요.
# ============================================================
CONFIG = {
    # 1. 경로 설정, 분리할 폴더 한단계 위의 폴더 선정.
    "DATA_PATH": r"C:\Users\소현우\Downloads\archive\DATASET\DATASET",
    "MODEL_NAME": "yolov8x-cls.pt",        # 학습 시작 시 사용할 기본 모델
    "PROJECT_NAME": "O_R_model",           # 결과가 저장될 폴더명
    
    # 2. 학습(Training) 하이퍼파라미터
    "EPOCHS": 10,
    "IMG_SIZE": 224,
    "BATCH_SIZE": 64,                      # GPU 메모리에 맞춰 조절 (32, 64, 128 등)
    "WORKERS": 4,                          # CPU 코어 사용량 (메모리 부족 시 줄임)
    
    # 3. 추론(Inference) 설정
    "WEIGHTS_PATH": "runs/classify/O_R_model/weights/best.pt",
    "USE_HALF": True,                      # FP16 반정밀도 가속 (RTX 50 시리즈 핵심)
    "WEBCAM_ID": 0                         # 연결된 웹캠 번호
}
# ============================================================

# 환경 설정 (CUDA 12.8 및 디바이스 체크)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if device == 'cuda:0':
    torch.backends.cudnn.benchmark = True
    print(f"🚀 CUDA 12.8 가속 활성화: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU 인식 실패. CPU로 구동합니다.")

def train_process():
    """모델 학습 프로세스"""
    model = YOLO(CONFIG["MODEL_NAME"])

    model.train(
        data=CONFIG["DATA_PATH"],
        epochs=CONFIG["EPOCHS"], 
        imgsz=CONFIG["IMG_SIZE"], 
        batch=CONFIG["BATCH_SIZE"],
        workers=CONFIG["WORKERS"],
        cache=False,
        device=0,
        name=CONFIG["PROJECT_NAME"],
        exist_ok=True
    )
    print("✅ 학습 완료")

def run_inference():
    """실시간 웹캠 추론 프로세스"""
    # 저장된 가중치를 불러와 GPU 메모리에 최적화 배치
    model = YOLO(CONFIG["WEIGHTS_PATH"]).to(device)

    print("📸 웹캠 시작 (종료하려면 'q'를 누르세요)")
    cap = cv2.VideoCapture(CONFIG["WEBCAM_ID"])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 추론 실행
        results = model.predict(
            frame, 
            device=device, 
            half=CONFIG["USE_HALF"],
            verbose=False
        )[0]

        # 결과값 파싱
        probs = results.probs
        label = results.names[probs.top1]
        conf = probs.top1conf.item()

        # UI 출력
        display_text = f"CUDA 12.8 | {label} {conf*100:.1f}%"
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("RTX 5060 & CUDA 12.8 TEST", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 실행하고자 하는 기능을 선택하세요.
    
    # 1. 학습 실행
    train_process()
    
    # 2. 추론 실행 (학습이 완료되어 best.pt가 생성된 후 아래 주석 해제)
    #run_inference()