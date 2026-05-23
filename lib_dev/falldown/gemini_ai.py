from google import genai
from PIL import Image
import cv2

class GeminiAnalyzer:

    def __init__(self, api_key):

        self.client = genai.Client(api_key=api_key)

    def analyze(self, frame):

        print("\n[시스템] Gemini 분석 시작")

        try:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            pil_img = Image.fromarray(rgb)

            prompt = (
                "이 이미지에 사람이 쓰러져 있는 상황인가요? "
                "위험 상태를 진단하고 환자의 자세와 "
                "주변 환경을 한국어로 간결하고 명확하게 "
                "설명해 주세요."
            )

            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=[pil_img, prompt]
            )

            print("\n[Gemini Response]")
            print(response.text)
            print("-" * 50)

        except Exception as e:

            print("Gemini 오류:", e)
