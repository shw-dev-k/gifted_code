import cv2
from detector import FallDetector
from ui import draw_ui
from config import CAMERA_INDEX

cap = cv2.VideoCapture(CAMERA_INDEX)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

ret, frame = cap.read()

if not ret:
    print("카메라를 읽을 수 없습니다.")
    exit()

h, w = frame.shape[:2]

cv2.namedWindow("Fall Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Fall Detection", w, h)

detector = FallDetector()

print("=" * 50)
print("실시간 낙상 감지 & Gemini 비전 연동 프로그램 시작")
print("=" * 50)

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    image, status = detector.process(frame)

    draw_ui(image, status)

    cv2.imshow("Fall Detection", image)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
