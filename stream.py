import streamlit as st
from pdf_reader import extract_text_from_pdf, chunk_text
from embedder import get_embeddings, build_faiss_index
from retriever import retrieve_relevant_chunks
from generator import generate_answer

st.set_page_config(page_title="PDF Q&A (RAG)", layout="wide")

st.title("RAG (Retrieval-Augmented Generation) Based Q&A ChatBot")

# File upload
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

if pdf_file:
    with st.spinner("Reading PDF..."):
        # Save the uploaded PDF
        with open("uploaded.pdf", "wb") as f:
            f.write(pdf_file.read())
        
        # Extract text and process
        text = extract_text_from_pdf("uploaded.pdf")
        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)
        index = build_faiss_index(embeddings)

        st.success("PDF processed! You can now ask questions.")

        query = st.text_input("Ask a question:")

        if query:
            with st.spinner("Retrieving and generating answer..."):
                relevant_chunks = retrieve_relevant_chunks(query, chunks, index)
                answer = generate_answer(relevant_chunks, query)
                st.markdown("### Answer:")
                st.info(answer)
