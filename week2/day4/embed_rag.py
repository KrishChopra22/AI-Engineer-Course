import os
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Sentence Transformer also has multiple models, no need of creating client
model = SentenceTransformer("all-MiniLM-L6-v2") #vector size 384

text = "Machine Learning is fun"

def cosine_similarity(a, b):
    return np.dot(a, b) / (
            np.linalg.norm(a) * np.linalg.norm(b)
        )

# embedding = model.encode(text)
# print(embedding.shape)  # tells size of vector
# print(embedding[:10]) # prints only first 10 array elements out of 384

text1 = "I love fruits"
text2 = "Lion eats deer"

v1 = model.encode(text1)    # Vector-1
v2 = model.encode(text2)    # Vector-2

print(cosine_similarity(v1, v2)) # between 1 to -1, 1 showing very similarity and -1 showing very dissimilarity