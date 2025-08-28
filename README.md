# 📚 RAG Q&A Bot

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions from your own documents.

## 🚀 Features
- Upload and query your own PDFs
- Answers grounded in your docs
- Uses ChromaDB for embeddings
- Works with local LLMs via Ollama

## 🛠️ Setup
```bash
git clone https://github.com/CodyMayers/rag-qa-bot.git
cd rag-qa-bot
python -m venv venv
venv\Scripts\activate # (mac/linux) source venv/bin/activate
pip install -r requirements.txt
```

You will also need to install [Ollama](https://ollama.com/), and then pull down the Mistral LLM using the below command:
```
ollama pull mistral
```

## ▶️ Usage
Add your documents to the data/ folder, then run the ingestion script:
```
python ingest.py
```
Launch the app:
```
streamlit run app.py
```

## ⚡ Roadmap
- Support multiple file formats (txt, md, docx)
- Show sources alongside answers
- Support loading all the files in the data folder at once rather than one at a time (while avoiding duplicates)
- UI element to allow user to limit the search to only some of the documents rather than all of them
