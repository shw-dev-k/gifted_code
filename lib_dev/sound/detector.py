from config import danger_words
from config import safe_words

# 위험 단어 검사
def is_danger(text):

    return any(
        word in text
        for word in danger_words
    )

# 안전 단어 검사
def is_safe(text):

    return any(
        word in text
        for word in safe_words
    )