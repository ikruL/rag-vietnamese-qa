# RAG Vietnamese QA Chatbot

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-green)](https://ollama.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A local RAG (Retrieval-Augmented Generation) chatbot built with **Ollama** + **ChromaDB** that answers questions based on the Vietnamese QA ver2 dataset (from Kaggle).

In purposes of learning RAG pipelines, local LLM inference, and Vietnamese NLP.
## Demo

<image-card alt="Demo" src="images/demo-chat.png" ></image-card> 

## Features
- Full RAG pipeline: Load dataset -> Chunking -> Embedding -> Store in ChromaDB -> Retrieve -> Chain -> Generate
- Good handling of paraphrase / similar questions (via prompt engineering)
- Embedding with `qwen3-embedding:0.6b`
- Using `"qwen3:4b` for LLM inference
- Easy to extend: Change dataset, embedding model, or add conversation memory

## Tech Stack
- **Language**: Python 3.10+
- **LLM & Embedding**: Ollama (local inference)
- **Vector Database**: ChromaDB (persistent)
- **Text splitting**: LangChain RecursiveCharacterTextSplitter
- **Data processing**: pandas, kagglehub

## Installation

1.  Clone the repository:
   
     ```bash
     git clone https://github.com/ikruL/rag-vietnamese-qa.git
     
     cd rag-vietnamese-qa
2. Create and activate virtual environment:
   
     ```bash
     python -m venv .venv
     
    .venv\Scripts\activate
3. Install dependencies:
     ```bash
     pip install -r requirements.txt
4. Install Ollama & pull models:
   - Download Ollama: https://ollama.com/download
   - Pull embedding & LLM models:
     
       ```bash
       ollama pull qwen3-embedding:0.6b
       
       ollama pull qwen3:4b
       
       ollama serve
5. Run the chatbot:
     ```bash
     python main.py

