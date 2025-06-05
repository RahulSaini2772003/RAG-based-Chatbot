from embedder import model

def retrieve_relevant_chunks(query, chunks, index, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]