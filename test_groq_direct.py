import os, requests, json
res = requests.post(
    "https://api.groq.com/openai/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.getenv('GROQ_API_KEY_1')}", "Content-Type": "application/json"},
    json={
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "messages": [
            {"role": "system", "content": "Extract 1 question-and-answer pair from the following text. Return ONLY a JSON object with 'q' and 'a' keys."},
            {"role": "user", "content": "TEXT:\nThis is a test of the emergency broadcast system. What is this a test of? It is a test of the emergency broadcast system."}
        ],
        "temperature": 0.1
    },
    timeout=30
)
print(res.status_code)
print(res.text)
