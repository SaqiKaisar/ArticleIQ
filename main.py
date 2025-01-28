import os
import streamlit as st
import pickle
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseLanguageModel
import time

from dotenv import load_dotenv

load_dotenv()

st.title("New Research Tool")
st.sidebar.title("News Article URLs")

if "urls" not in st.session_state:
    st.session_state.urls = [""]
if "no_of_urls" not in st.session_state:
    st.session_state.no_of_urls = 1


def add_url_field():
    global no_of_urls
    st.session_state.urls.append("")
    st.session_state.no_of_urls += 1


def create_qa_chain(
    llm: BaseLanguageModel, vector_store: VectorStore
) -> RetrievalQAWithSourcesChain:
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, retriever=vector_store.as_retriever()
    )
    return chain


if st.sidebar.button("Add another URL"):
    if st.session_state.no_of_urls < 5:
        add_url_field()

for i, url in enumerate(st.session_state.urls):
    st.session_state.urls[i] = st.sidebar.text_input(f"URL {i+1}", value=url)

process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_vector_store.pkl"
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
main_placeholder = st.empty()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=st.session_state.urls)
    main_placeholder.text(f"Data Loading...")
    data = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ",S"], chunk_size=1000
    )
    main_placeholder.text(f"Text spitting...")
    docs = splitter.split_documents(data)
    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    main_placeholder.text("Embedding Vector Started Building...")
    vector_store = FAISS.from_documents(docs, embeddings)
    time.sleep(2)
    with open(file_path, "wb") as f:
        pickle.dump(vector_store, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_store = pickle.load(f)
            chain = create_qa_chain(llm, vector_store)
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])
