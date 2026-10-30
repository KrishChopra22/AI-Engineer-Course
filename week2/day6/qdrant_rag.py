# IMPORTS
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq

# LOAD ENV VARIABLES
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# CONNECT TO QDRANT
client = QdrantClient(url = QDRANT_URL, api_key = QDRANT_API_KEY)
print("Connected to Qdrant Cloud")

# CREATE QDRANT COLLECTION (inside Vector DB)
COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384

# Delete collection if it already exists
if client.collection_exists(COLLECTION_NAME):
    print(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)

# Creating collection
client.create_collection(
    collection_name = COLLECTION_NAME,
    vectors_config = VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
)
print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")

# LOAD OUR KNOWLEDGE
with open("knowledge.txt", "r", encoding="utf-8") as f:
    documents = [ # add line to document if not null
        line.strip()
        for line in f
        if line.strip()
    ]
print(f"Loaded {len(documents)} documents")

# CREATE EMBEDDINGS
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")     # vector size 384
print("Embedding model is ready!")

embeddings = model.encode(documents)

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding size: {len(embeddings[0])}")

# CREATE QDRANT POINTS (inside collection)
points = []
for i, embedding in enumerate(embeddings):
    point = PointStruct(
        id = i+1,
        vector = embedding.tolist(),
        payload = {
            "text" : documents[i]
        }
    )
    points.append(point)

# UPLOAD this knowledge base TO QDRANT
client.upsert(      # upsert = upload + insert (check if point not exist then only upload)
    collection_name=COLLECTION_NAME,
    points=points
)
print(f"Uploaded {len(points)} Documents to Qdrant")

# SEARCH & MATCH IN QDRANT
def search(query, top_k=3):
    # first, convert query into embedding
    query_vector = model.encode(query).tolist()

    # search Qdrant for similar vectors
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,    # matches top 3 cosine similarity
        with_payload=True,
    ).points

    return results      # returns top 3 points from collection

# TEST SEARCHING IN YOUR QDRANT

query = "Will I also get a dinner at my office after work?"
search_results = search(query, top_k=3)

print("\nSearch Results :")

for result in search_results:
    print(f"Score : {result.score:.3f}")
    print(result.payload["text"])
    print()

# CONNECT TO GROQ
groq_client = Groq(api_key = GROQ_API_KEY)
groq_model = "openai/gpt-oss-120b"

# ASK THE LLM
def ask_llm(question, context):
    
    system_message = {
        "role": "system",
        "content": f"""
        Answer the question using only the information provided below.
        
        CONTEXT: {context} 
        
        If the answer is completely unrelated to the context, say I can't answer this question based on the information provided."""
    }
    message={
        "role": "user",
        "content": question
    }
    messages = [system_message, message]
    response = groq_client.chat.completions.create(model=groq_model, messages=messages)
    answer = response.choices[0].message.content
    return answer


# COMPLETE RAG PIPELINE : combine all steps

question = "Can I work from my office on Monday?"

results = search_results

context = "\n".join(
    result.payload["text"]
    for result in results
)
final_answer = ask_llm(question, context)

print(f"\nYour Question : \n{question}\n")
print(f"Final Answer : \n{final_answer}")