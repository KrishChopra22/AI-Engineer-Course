import os
import time
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

# PART-1 : JOB SCHEMA

job_description = f"""
Out of the successful launch of Chase in 2021, we're a new team, with a new mission. We're creating products that solve real world problems and put customers at the center - all in an environment that nurtures skills and helps you realize your potential. Our team is key to our success. We're people-first. We value collaboration, curiosity and commitment. 

As a Lead Software Engineer at JPMorgan Chase within the Accelerator business, you are the heart of this venture, focused on getting smart ideas into the hands of our customers. You have a curious mindset, thrive in collaborative squads, and are passionate about new technology. By your nature, you are also solution-oriented, commercially savvy and have a head for fintech. You thrive in working in tribes and squads that focus on specific products and projects - and depending on your strengths and interests, you'll have the opportunity to move between them.

While we're looking for professional skills, culture is just as important to us. We understand that everyone's unique - and that diversity of thought, experience and background is what makes a good team, great. By bringing people with different points of view together, we can represent everyone and truly reflect the communities we serve. This way, there's scope for you to make a huge difference - on us as a company, and on our clients and business partners around the world.

Job Responsibilities:

Use domain modeling techniques to build best-in-class business products.
Structure software for ease of understanding, testing, and evolution.
Build scalable solutions that avoid single points of failure.
Develop secure code to protect customers and the company from malicious actors.
Investigate and resolve issues promptly, ensuring they do not resurface.
Ensure zero downtime for end-users during releases.
Optimize data writing and reading for our needs.
Monitor performance and identify solutions to problems.
Ensure systems are reliable and easy to operate.
Continuously update technologies and patterns to keep us current.
Required Qualifications, Capabilities, and Skills:

Formal training or certification in software engineering concepts with 5+ years of applied experience.
Experience with JVM-based languages (Java/Kotlin) and cloud-native solutions.
Familiarity with open-source technologies and multi-cloud solutions.
Ability to independently assess problems and select appropriate technologies.
Willingness to explore new solutions while focusing on long-lasting, real-world problem-solving.
Desire to teach and share knowledge, valuing team collaboration over individual heroics.
Ability to adjust quickly to changing priorities and conditions.
Long-term vision, choosing technologies and approaches based on end goals.
High standards for personal and team performance.
Drives team adoption of enterprise-authorized AI-assisted engineering practices within the work environment to improve code quality, delivery speed, and operational outcomes (e.g., AI-assisted code review/refactoring, test strategy acceleration, incident/root-cause analysis support), while establishing consistent validation standards (secure coding, peer review, automated testing) and promoting reuse of effective patterns across the team.
Applies knowledge of tools within the Software Development Life Cycle toolchain, including enterprise-authorized AI-assisted development and automation capabilities, to improve the value realized by automation
 

Preferred Qualifications, Capabilities, and Skills:

Initiative to shape your own path with a pragmatic and iterative approach.
Ethical integrity, earning and maintaining trust.
Demonstrated experience leading effective use of approved AI-assisted software development tools (e.g., for coding, code review, test acceleration, troubleshooting) with the ability to set team expectations for validating AI outputs for correctness, performance, and security.
Strong understanding of responsible AI use in engineering workflows, including data sensitivity considerations, secure handling of inputs/outputs, and adherence to resiliency and security expectations; experience coaching engineers on safe, compliant adoption within delivery practices
"""

class JobDescription(BaseModel):
    role: str
    req_skills: list[str]
    pref_skills: list[str]
    min_experience: float | None
    education_requirements: list[str]
    responsibilities: list[str]

job_schema = JobDescription.model_json_schema()

response_format = {
    "type": "json_object"
}

def parse_job_description(job_description):
    system_prompt = f"""
    You're an expert skilled HR assistant.
    Your job is to analyze the job description and extract the structured information from them, return only valid JSON matching this schema {job_schema}.
    Important:
    DO NOT return unnecessary fields, neither return empty schema, fill the schema with actual information extracted from the job description. If minimum experience is not specified return null and any information for any field is missing return empty list.
    DO NOT invent information
    """
    user_prompt = f"""
    Analyze the following Job Description
    {job_description}
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

    response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content

    # read this json output
    job_data = json.loads(raw_output)
    jobD = JobDescription(**job_data)

    return jobD

# PART-2 : RESUME SCHEMA

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

class MatchResult(BaseModel):
    score: float
    details: dict
match_schema = MatchResult.model_json_schema()

# PART-3 : PARSE RESUME


def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# def read_docx(file_path):
#     document = Document(file_path)
#     text = ""
#     for paragraph in document.paragraphs:
#         if paragraph.text.strip():
#             text += paragraph.text + "\n"

#     for table in document.tables:
#         for row in table.rows:
#             for cell in row.cells:
#                 if cell.text.strip():
#                     text += cell.text + "\n"
#     return text

def read_resume(file_path):  # To check whether PDF or DOC
    if file_path.suffix.lower() == ".pdf":
        return read_pdf(file_path)
    if file_path.suffix.lower() == ".docx":
        return read_docx(file_path)
    else:
        return None

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

# PART-4 : MATCHING RESULTS (SCORE)
def final_score(job, resume):
    user_prompt = f"""
    You are an expert HR recruiter.
    Compare the candidate's resume with the job dscription.

    Job Description:
    {job.model_dump_json(indent=2)}
    
    Cadidate Resume:
    {resume.model_dump_json(indent=2)}

    Return result in JSON matching this schema {match_schema}

    Give me the following info in Details:
    1. Candidate Name
    2. Matching Skills
    3. Missing Important Skills
    4. Whether experience requirement is met
    5. Overall match percentage from 0 to 100
    6. A short final verdict

    Keept the response concise and easy to read.
    """
    user_message = {
        "role": "user",
        "content": user_prompt
    }
    messages = [user_message]
    response = client.chat.completions.create(model=model, messages=messages, response_format=response_format)
    raw_output = response.choices[0].message.content
    result_data = json.loads(raw_output)
    final_result = MatchResult(**result_data)
    return final_result


# PARSING RESUMES
resume_folder = Path("resumes")
all_results = []
parsed_jobD = parse_job_description(job_description)
for file_path in resume_folder.iterdir():
    if file_path.suffix.lower() not in [".pdf", ".docx"]:
        continue
    print("\nProcessing: ", file_path.name)
    resume_text = read_resume(file_path)
    parsed_resume = parse_resume(resume_text)
    time.sleep(5)
    result = final_score(parsed_jobD, parsed_resume)
    time.sleep(5)
    print("Score: ", result.score)
    all_results.append({
        "name": parsed_resume.name,
        "score": result.score,
        "details": result.details
    })
all_results.sort(key=lambda candidate : candidate["score"], reverse=True)
top_2 = all_results[:2]
worst_2 = all_results[-2:]