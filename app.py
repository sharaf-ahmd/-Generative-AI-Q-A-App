# streamlit_app.py
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize LLM (v0.3.27 style)
llm = OpenAI(temperature=0, openai_api_key=api_key)

# Load the website as a document
url = "https://www.geeksforgeeks.org/artificial-intelligence/what-is-generative-ai/?"
loader = WebBaseLoader(url)
docs = loader.load()

# Optional: clean docs by keeping only page_content
clean_docs = [Document(page_content=doc.page_content) for doc in docs]

# Create a vector store for retrieval
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
db = FAISS.from_documents(clean_docs, embeddings)
retriever = db.as_retriever()

# Create a prompt template and document chain
prompt = ChatPromptTemplate.from_template("""
Answer the question **in a clear and concise format** based on the context below.
- Only include relevant information.
- Remove menus, ads, or unrelated text.
- Keep paragraphs short.

<context>
{context}
</context>

Question: {input}
""")

doc_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain
ret_chain = create_retrieval_chain(retriever, doc_chain)

# Streamlit UI
st.title("Generative AI Q&A App")
st.write("Ask questions about Generative AI based on GeeksforGeeks article!")

question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Fetching answer..."):
            # Invoke the retrieval chain
            result = ret_chain.invoke({"input": question})
            
            # Ensure the answer is a clean string
            answer = result.get("output_text") if isinstance(result, dict) else str(result)
            
            st.success("Answer:")
            st.write(answer)
