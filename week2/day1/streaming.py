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

prompt = "Explain how internet works."
message = {
    "role": "user",
    "content": prompt
}
messages = [message]

# response = client.chat.completions.create(model=model, messages=messages)     #Response not Stream
# answer = response.choices[0].message.content
stream = client.chat.completions.create(model=model, messages=messages, stream=True)  # Stream (chunk by chunk)

for chunk in stream:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
