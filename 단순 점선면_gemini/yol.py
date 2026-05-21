import cv2
from ultralytics import YOLO

# =========================
# 학습
# =========================

model = YOLO("yolov8x-cls.pt")

model.train(
    data=r"C:\Users\sygzz\Downloads\archive\DATASET\DATASET",

    epochs=50,

    imgsz=224,

    batch=32,

    workers=8,

    cache=True,

    verbose=True,

    name="O_R_model",

    exist_ok=True
)

print("학습 완료")

# =========================
# 학습된 모델 불러오기
# =========================

model = YOLO("runs/classify/O_R_model/weights/best.pt")

print("웹캠 시작")

# 외부 웹캠이면 1
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        print("카메라 오류")
        break

    # 예측
    results = model.predict(frame, verbose=False)[0]

    probs = results.probs

    top1 = probs.top1
    conf = probs.top1conf.item()

    label = results.names[top1]

    # 출력 텍스트
    text = f"{label} {conf*100:.1f}%"

    # 화면에 표시
    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("AI Camera", frame)

    # q 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()