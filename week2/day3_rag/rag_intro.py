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

# Step-1
knowledge_base = {
    "age": "The age of Krish Chopra is 24 years",
    "Net Worth": "Krish's net worth is 96 Lacs",
    "colour": "Krish Chopra's favourite colour is Black"
}

# Step-2
def retrieve_info(question):
    question = question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "net worth" in question:
        return knowledge_base["Net Worth"]
    elif "colour" in question:
        return knowledge_base["colour"]
    else:
        return None

def ask_llm(question):
    # Step-3
    context = retrieve_info(question)

    system_message = {
        "role": "system",
        "content": f"""Answer in one or two lines based on this context only
        CONTEXT: {context}"""
    }
    message={
        "role": "user",
        "content": question
    }
    messages = [system_message, message]
    response = client.chat.completions.create(model=model, messages=messages)
    answer = response.choices[0].message.content
    return answer

question = "How old is Krish"   # This shows rigidness, faulty
print(ask_llm(question))