import cv2
from google import genai
from PIL import Image
import io

# 1. Gemini 클라이언트 설정
# (보안을 위해 API 키는 환경변수로 등록하는 것을 추천하지만, 테스트를 위해 유지합니다)
client = genai.Client(api_key="your_api_key_here")  # 실제 API 키로 교체하세요

# 2. 웹캠 연결 (0번은 컴퓨터에 연결된 기본 카메라입니다)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. 연결을 확인해 주세요.")
    exit()

print("=" * 50)
print("실시간 카메라 연동 프로그램이 시작되었습니다.")
print("- [Space바]: 현재 화면을 캡처해서 Gemini에게 질문하기")
print("- [q 키]: 프로그램 종료")
print("=" * 50)

while True:
    # 카메라로부터 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("화면을 가져올 수 없습니다.")
        break

    # 화면에 카메라 영상 표시
    cv2.imshow('Gemini Vision Cam (Press SPACE to Ask, Q to Quit)', frame)

    # 키 입력 대기
    key = cv2.waitKey(1) & 0xFF

    # 1. Space바를 누르면 Gemini에게 이미지와 함께 질문 전송
    if key == ord(' '):
        print("\n[시스템] 현재 화면을 캡처 중...")
        
        # OpenCV 이미지(BGR)를 Gemini가 인식할 수 있는 형태로 변환 (BGR -> RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # 캡처 당시 상황에 대해 물어볼 질문 입력 (텍스트를 고정하거나 input으로 받을 수도 있습니다)
        prompt = "이 이미지에 무엇이 보이나요? 한국어로 친절하게 설명해 주세요."
        
        print(f"[나]: {prompt}")
        print("[Gemini]: 생각 중...")

        try:
            # gemini-2.5-flash-lite 모델은 멀티모달(텍스트+이미지)을 지원합니다.
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[pil_img, prompt]
            )
            print(f"[Gemini Response]:\n{response.text}\n")
            print("-" * 50)
        except Exception as e:
            print(f"에러 발생: {e}")

    # 2. 'q'를 누르면 루프 종료
    elif key == ord('q'):
        print("[시스템] 프로그램을 종료합니다.")
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()