from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk

nltk.download('punkt')
qa_model = pipeline("text2text-generation", model="google/flan-t5-large") 

def simple_sent_tokenize(text):
    return [s.strip() for s in text.split('.') if s.strip()]

def preprocess_context(context_chunks):
    context = " ".join(context_chunks)
    sentences = simple_sent_tokenize(context)
    return " ".join(sentences[:20])


def generate_answer(context_chunks, question):
    context = preprocess_context(context_chunks)
    
    prompt = f"""You are a highly knowledgeable assistant. Carefully read the provided context and generate an accurate, structured, and detailed answer to the question. If the context does not contain enough information, say 'The answer is not available in the provided context.'\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
    
    answer = qa_model(prompt, max_new_tokens=256, truncation=True)[0]["generated_text"]
    return answer.strip()
