import speech_recognition as sr

recognizer = sr.Recognizer()

mic = sr.Microphone()

# 인식률 설정
recognizer.energy_threshold = 200
recognizer.dynamic_energy_threshold = True
recognizer.pause_threshold = 1.0

# -----------------------------
# 최초 1번만 소음 측정
# -----------------------------
with mic as source:

    print("초기 소음 측정 중...")

    recognizer.adjust_for_ambient_noise(
        source,
        duration=2
    )

def listen_text(timeout=None):

    with mic as source:

        print("듣는 중...")

        audio = recognizer.listen(
            source,
            timeout=timeout,
            phrase_time_limit=6
        )

    text = recognizer.recognize_google(
        audio,
        language="ko-KR"
    )

    return text.strip(), audio