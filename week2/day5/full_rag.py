import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API Key Error")

client = Groq(api_key = my_api_key)

groq_model = "openai/gpt-oss-120b"
embed_model = SentenceTransformer("all-MiniLM-L6-v2") #vector size 384

# Knowledge Base
documents = [
    "Employees get 24 days of paid leaves per year",
    "Employees work from office on Tuesdays, Wednesdays & Thursdays",
    "Mondays and Fridays are optional work from home days.",
    "Employees have a 60 days notice period",
    "Employees can claim Rs. 2500 per month for home internet/wifi recharge",
    "Employees get lunch and high-tea at office"
]

document_embeddings = embed_model.encode(documents)     # list of vector knowledge base

# to find similarity between 2 vectors
def cosine_similarity(a, b):
    return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

# to retrieve each similarity score (NON SCALABLE METHOD)
def retrieve(query_embed):
    scores = []
    for i, doc_embed in enumerate(document_embeddings):
        score = cosine_similarity(query_embed, doc_embed)
        scores.append((score, documents[i]))     # append both score and it's respective document line

    scores.sort(reverse=True)   # highest matching score will be at top
    return scores[0]    # it will return both score & doc

def ask_llm(question, context):
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
    response = client.chat.completions.create(model=groq_model, messages=messages)
    answer = response.choices[0].message.content
    return answer

query = "Today's wednesday, can I stay at home and finish my work?"
query_embedding = embed_model.encode(query)
score, context = retrieve(query_embedding)

print(ask_llm(query, context))