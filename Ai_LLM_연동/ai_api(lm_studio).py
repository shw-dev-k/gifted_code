# 이거 실행해서 token/s 확인
import requests, time

start = time.time()
res = requests.post("http://localhost:1234/v1/chat/completions",
    json={
        "model": "gemma-3-4b-instruct",
        "messages": [{"role": "user", "content": "안녕"}],
        "stream": False
    }
).json()

elapsed = time.time() - start
tokens = res["usage"]["completion_tokens"]
print(f"{tokens / elapsed:.1f} tokens/s")