import os
from pathlib import Path

# donot use name as token.py as load_dotenv uses same file
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API Key Error")

client = Groq(api_key = my_api_key)

# model = "llama-3.3-70b-versatile"     # model not working
model = "openai/gpt-oss-120b"

prompt1 = "Hi!"
prompt2 = "Explain time travel in detail"
prompt3 = "Write an essay in 1000 words on Artificial Intelligence's future "

prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    message = {
        "role": "user",
        "content": prompt
    }

    messages = [message]
    
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=100)   # maximum tokens is 100
    usage = response.usage
    print(f"Prompt : {prompt} \n--> Prompt Tokens : {usage.prompt_tokens} & Response Tokens : {usage.completion_tokens} \n--> Finish Reason : {response.choices[0].finish_reason}")

# print(response.choices[0].message.content)
