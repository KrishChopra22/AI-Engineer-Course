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
content = "Do you know who is Krish Chopra?"

message = {
    "role": role,
    "content": content
}

messages = [message]

response = client.chat.completions.create(model=model, messages=messages)
print(response)