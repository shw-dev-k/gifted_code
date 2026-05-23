from google import genai

client = genai.Client(api_key="AIzaSyA0VvM6ZZZUiUT615e3NTymacwpY--1vuk")

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents="안녕하세요. 현재 vs code에서 진행중입니다."
)
print(response.text)

#https://ai.google.dev/gemini-api/docs/pricing?hl=ko#1_5flash