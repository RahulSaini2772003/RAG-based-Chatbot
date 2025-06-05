from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embeddings(chunks):
    return model.encode(chunks)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index