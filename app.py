# streamlit_app.py
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langchain.chat_models import OpenAI

load_dotenv()


llm = OpenAI(temperature=0) 
api_key = os.getenv("OPENAI_API_KEY")



# Load the website as a document

url = "https://www.geeksforgeeks.org/artificial-intelligence/what-is-generative-ai/?"
loader = WebBaseLoader(url)
docs = loader.load()  


# Create a vector store for retrieval
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()


# Create a prompt template and document chain
prompt = ChatPromptTemplate.from_template("""
Use the context below to answer the question. 
If the answer is not in the context, say "I don't know."

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
            answer = ret_chain.invoke({"input": question})
            st.success("Answer:")
            st.write(answer)
