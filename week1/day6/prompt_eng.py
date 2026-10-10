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

def llm_ans(prompt):
    message={
        "role": "user",
        "content": prompt
    }
    messages = [message]
    response = client.chat.completions.create(model=model, messages=messages)
    ans = response.choices[0].message.content
    return ans

bad_prompt = """
This is a user complaint: My laptop is not working.
Classify this.
"""

good_prompt = """
#ROLE:
You are an expert customer support executive at a mobile/laptop company.
#TASK:
You have to classify the issue / user complaint in a category
#CONSTRAINT:
You have to classify the issue in one of the 3 categories namely : Billing, Technical, Return
#OUTPUT FORMAT:
Your answer should be in one word only, one word should be a category given in constraints.
#EXAMPLE:
For instance if a user says he want a refund then the category is - Refund
#FALLBACK:
If the issue is unrealated to any of the categories mentioned in the constraints, then the answer should be OTHER

This is a user complaint:
My laptop is not working.
"""

print(llm_ans(good_prompt))