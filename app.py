from pdf_reader import extract_text_from_pdf, chunk_text
from embedder import get_embeddings, build_faiss_index
from retriever import retrieve_relevant_chunks
from generator import generate_answer

pdf_path = "sample.pdf"

print("[INFO] Extracting text from PDF...")
text = extract_text_from_pdf(pdf_path)

print("[INFO] Chunking text...")
chunks = chunk_text(text)

print("[INFO] Generating embeddings...")
embeddings = get_embeddings(chunks)

print("[INFO] Building FAISS index...")
index = build_faiss_index(embeddings)

while True:
    query = input("\nAsk a question (or type 'exit'): ")
    if query.lower() == "exit":
        break

    print("[INFO] Retrieving relevant chunks...")
    relevant_chunks = retrieve_relevant_chunks(query, chunks, index)

    print("[INFO] Generating answer...")
    answer = generate_answer(relevant_chunks, query)

    print("\n🧠 Answer:", answer)