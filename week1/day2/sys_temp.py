import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API Key Error")

client = Groq(api_key = my_api_key)

# model = "llama-3.3-70b-versatile"     # model not working
model = "openai/gpt-oss-120b"

role = "user"
# content = "Hi you're looking pretty today, I love you."
content = "Suggest a name for my clothing company."

# This shows system-role
# message_system = {
#     "role": "system",
#     "content": "You are my strict office colleague."
# }

message_system = {
    "role": "system",
    "content": "You are a brand manager who suggests one word name"
}

message = {
    "role": role,
    "content": content
}

messages = [message_system, message]

# By default temperature is 0, means safe play, range is [0,2]
response = client.chat.completions.create(model=model, messages=messages, temperature=2)
answer = response.choices[0].message.content
print(answer)