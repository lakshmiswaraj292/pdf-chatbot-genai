import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="Chat with Your Notes", layout="centered")

st.title("📄 Chat with Your Notes - PDF Q&A Bot")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

OPENAI_API_KEY = st.text_input("Enter your OpenAI API Key", type="password")

if uploaded_file and OPENAI_API_KEY:
    with st.spinner("Processing your PDF..."):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(docs, embeddings)

        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())

        st.success("PDF loaded. You can now ask questions!")

        user_query = st.text_input("Ask a question about your PDF:")
        if user_query:
            response = qa_chain.run(user_query)
            st.write("🤖 Answer:", response)