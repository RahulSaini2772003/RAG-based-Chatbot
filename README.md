# PDF-based Question Answering System

This project allows users to ask questions based on the content of a PDF document using semantic search and generative models.

## Features

- Extracts text from PDFs
- Chunks and embeds text using sentence-transformers
- Retrieves relevant chunks with FAISS
- Generates answers using Flan-T5 model

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

```
pdfplumber
faiss-cpu
sentence-transformers
transformers
torch
```

## File Structure

```
.
├── extract.py
├── embedder.py
├── retriever.py
├── generator.py
├── main.py
├── requirements.txt
└── README.md
```

## License

MIT License
