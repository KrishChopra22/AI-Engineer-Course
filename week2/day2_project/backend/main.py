from fastapi import FastAPI     # FastAPI creates application
# uvicorn runs this application

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel
import json
from pypdf import PdfReader
# from docx import Document

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API Key Error")

client = Groq(api_key = my_api_key)

# model = "llama-3.3-70b-versatile"     # model not working
model = "openai/gpt-oss-120b"


app = FastAPI()

def read_pdf(file_path: Path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

class WorkExperience(BaseModel):
    company: str | None = None  # optional field
    role: str | None = None
    duration: float | None = None
    description: str | None = None
    skills_used: list[str] = []

class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_exp_years: float | None = None
    skills: list[str] = []
    experiences: list[WorkExperience] = []
    education: list[str] = []
    projects: list[str] = []
resume_schema = Resume.model_json_schema()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str


def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.
    Extract the information from the resume based on it's meaning not only based on exact section headings.
    For example, 
    - Experience
    - Professional Experience
    - Work Experience / History
    - Employments
    - Internships

    These all may contain relevant experience. Skills may also appear in the skills section, work experience, projects.

    Return only valid JSON matching this schema.
    {resume_schema}

    Important Rules :-
    1. Do not invent any information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """

    user_prompt = f"""
    Parse the following resume:
    {resume_text}
    """

    system_message = {
        "role": "system",
        "content": system_prompt
    }
    user_message = {
        "role": "user",
        "content": user_prompt
    }
    messages = [system_message, user_message]
    response_format = {
        "type": "json_object"
    }

    response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content
    resume_data = json.loads(raw_output)
    resume = Resume(**resume_data)
    return resume

def ask_candidate(question: str, resume: Resume):
    system_prompt = f"""
    You are an AI assistant representing a job candidate. 
    Below is everything you know about the candidate.

    {resume.model_dump_json(indent=2)}

    Rules:
    1. Answer only using this information
    2. Never hallucinate
    3. If information is unavailable try to analyze based on given related information to you, else simply say
    "I don't have enough information to answer that"
    4. Be professional
    5. Answer as if any HR is interviewing this candidate.
    """
    system_message = {
            "role": "system",
            "content": system_prompt
        }
    user_message = {
            "role": "user",
            "content": question
        }
    messages = [system_message, user_message]
        
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content    

@app.get("/")   # Root of your Application
def home():
    return {
        "message": "HireMeAI is running!"
    }

@app.post("/chat")
def chat(request: ChatRequest):
    resume_text = read_pdf(Path("Ashish_Resume_final.pdf"))
    resume = parse_resume(resume_text)    # resume parsed
    # print(resume.model_dump_json(indent=2))

    answer = ask_candidate(request.question, resume)
    return {
        "answer": answer
    }