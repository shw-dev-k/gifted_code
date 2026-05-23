import time
import speech_recognition as sr

from speech import speak
from detector import is_danger, is_safe
from listener import listen_text

print("프로그램 시작")
print("'도와줘'라고 말하면 작동합니다.")

speak("응급 감지 시스템이 시작되었습니다.")

time.sleep(2)


while True:

    try:

      
        text, audio = listen_text()

        print("인식된 말:", text)

       
        if is_danger(text):

            print("위험 단어 감지")

          
            with open("temp.wav", "wb") as f:

                f.write(audio.get_wav_data())

            emergency = True

           
            for attempt in range(2):

                speak("괜찮으십니까?")

                time.sleep(2)

                try:

                 
                    response, response_audio = listen_text(
                        timeout=10
                    )

                   
                    with open(
                        "response.wav",
                        "wb"
                    ) as f:

                        f.write(
                            response_audio.get_wav_data()
                        )

                    print("응답:", response)

                    if is_safe(response):

                        emergency = False
                        break

                  
                    elif is_danger(response):

                        emergency = True
                        break

               
                    else:

                        if attempt == 0:

                            speak(
                                "다시 한 번 말씀해 주세요."
                            )

             
                except sr.WaitTimeoutError:

                    print("응답 시간 초과")

                    if attempt == 0:

                        speak(
                            "다시 한 번 말씀해 주세요."
                        )

           
                except Exception as e:

                    print("응답 인식 실패:", e)

                    if attempt == 0:

                        speak(
                            "다시 한 번 말씀해 주세요."
                        )

      
            if emergency:

                speak(
                    "전화로 도움을 요청하겠습니다."
                )

                time.sleep(5)

            else:

                speak("다행이군요")

                time.sleep(3)

    except Exception as e:

        print("오류 발생:", e)