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
text = "Hello I'm Ajay, I live in Delhi. I am currently facing lag issue with my iPhone 16 Pro, it shuts down abruptly and lags a lot. My email is ajay@gmail.com and my contact number is 9826103242."
content = f"""
This is customer ticket, please ectract the person's information from this {text}
"""

message = {
    "role": role,
    "content": content
}

# without pydantic output is unstructured
from pydantic import BaseModel

class Ticket(BaseModel):
    name:str
    email:str
    issue:str

schema = Ticket.model_json_schema()

response_format = {
    "type": "json_object"
}

system_prompt = f"""
Extract the personal information from the ticket strictly based on this schema {schema} and give a json output.
"""

system_message = {
    "role": "system",
    "content": system_prompt
}


messages = [system_message, message]

response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
answer = response.choices[0].message.content
# print(answer)

# read this json output
import json
raw_json = answer
data_file = json.loads(raw_json)
ticket = Ticket(**data_file)

print(ticket.email)