# Generative AI Q&A App

A web application that allows users to ask questions about **Generative AI** and get answers using information extracted from a GeeksforGeeks article. Built using **Streamlit**, **LangChain**, **OpenAI LLMs**, and **FAISS** for retrieval-based question answering.

---

## 🚀 Features

- Ask questions about **Generative AI** and receive concise, relevant answers.
- Uses **Web Scraping** to fetch content from a trusted educational source (GeeksforGeeks).
- Employs **LangChain Document & Retrieval Chains** for semantic search.
- Vector-based semantic search with **FAISS** and **OpenAI embeddings**.
- Interactive and simple **Streamlit UI**.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Streamlit** – Web application interface
- **LangChain v0.3.27** – LLM chains (DocumentChain & RetrievalChain)
- **FAISS** – Vector store for semantic search
- **OpenAI API** – LLMs and embeddings
- **WebBaseLoader** – Web scraping for document loading
- **dotenv** – Manage API keys securely

---


genai-qa-app/
│
├─ streamlit_app.py       # Main Streamlit application
├─ README.md              # Project documentation
├─ requirements.txt       # Python dependencies (optional)
└─ .env                   # Stores API keys securely

---

## 🔧 Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/genai-qa-app.git
cd genai-qa-app
---


