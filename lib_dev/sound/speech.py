import pyttsx3
import time

# 음성 엔진 설정
engine = pyttsx3.init()

engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

# 한국어 음성 선택
voices = engine.getProperty('voices')

for voice in voices:

    if (
        "korean" in voice.name.lower()
        or "heami" in voice.name.lower()
        or "ko_" in voice.id.lower()
    ):

        engine.setProperty('voice', voice.id)
        break

# 말하기 함수
def speak(text):

    print("컴퓨터:", text)

    engine.stop()

    engine.say(text)

    engine.runAndWait()

    time.sleep(0.5)