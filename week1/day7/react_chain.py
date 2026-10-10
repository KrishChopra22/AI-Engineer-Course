import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import re, time

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API Key Error")

client = Groq(api_key = my_api_key)

# model = "llama-3.3-70b-versatile"     # model not working
model = "openai/gpt-oss-120b"

# Tools (APIs)
def get_product_price(product):
    if product == "iPhone 17":
        return 8000
    if product == "iPhone 15":
        return 5000
    else:
        return 0

def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Calc error!"

tools = {
    "get_product_price": get_product_price,
    "calculator": calculator
}

#ReACT strategy
system_prompt = """
You are a Shopping Assistant.
You have these tools:

get_product_price(product)
calculator(expression)

IMPORTANT:
Call tools exactly like these examples:-

Action: get_product_price("iPhone 17")
Action: calculator("7000 - 2000")

Never write:
get_product_price(product="iPhone 17")
Never write:
calculator(expression="7000 - 2002")

Follow these rules:
1. Decide what you need to do next.
2. Call only one tool at a time.
3. After writing an action, STOP immediately.
4. Never guess or invent a tool result.
5. Wait until you receive an Observation.
6. Then decide your next action.
7. When the task is complete give the final answer.

Response Format:-
Thought: what you need to do
Action: tool_name(argument)

When finished:
Final Answer: your anser
"""
def run_agent(question):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]

    for step in range(5):
        print("\n--------------")
        print("STEP", step+1)
        print("--------------")

        response = client.chat.completions.create(model=model, messages=messages, temperature=0)
        answer = response.choices[0].message.content
        print(answer)

        # Agent ahs finished
        if "Final Answer:" in answer:
            break

        # Find the action
        match = re.search(
            r"Action:\s*(\w+)\((.*?)\)",
            answer
        )

        if match:
            tool_name = match.group(1)
            tool_input = match.group(2)
            tool_input = tool_input.strip()
            tool_input = tool_input.strip('"')

            # Run the Tool
            if tool_name in tools:
                tool = tools[tool_name]
                observation = tool(tool_input)

            else:
                observation = "Tool Not Found"

            print(
                "Observation:", observation
            )

            # Add LLM response to memory
            messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            # Give tool result (observation) back to LLM
            messages.append({
                "role": "user",
                "content": "Observation: " + str(observation)
            })
            time.sleep(5)

query = """
I have 8400 rs. What is the price of iPhone 17?
How much money will I be left with if I buy it right now.
"""
run_agent(query)