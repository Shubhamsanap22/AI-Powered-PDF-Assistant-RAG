import os
import hashlib
import tempfile
import shutil

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="centered")

CHROMA_ROOT = "chroma_db"  # each uploaded file gets its own subfolder here


# ---------- Helpers ----------
def get_file_hash(file_bytes: bytes) -> str:
    """Unique id for the uploaded file so we don't rebuild the DB every rerun."""
    return hashlib.md5(file_bytes).hexdigest()[:12]


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return MistralAIEmbeddings(model="mistral-embed")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")


def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )


@st.cache_resource(show_spinner=False)
def build_vectorstore(file_hash: str, tmp_pdf_path: str):
    """Load the PDF, split it, embed it, and persist/load a Chroma DB for this file."""
    persist_dir = os.path.join(CHROMA_ROOT, file_hash)

    embedding_model = get_embedding_model()

    if os.path.exists(persist_dir):
        # Already processed this exact file before — just load it
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
    else:
        loader = PyPDFLoader(tmp_pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )

    return vectorstore


def get_answer(retriever, llm, prompt, query: str) -> str:
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content


# ---------- Session state ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_file" not in st.session_state:
    st.session_state.active_file = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None


# ---------- Sidebar: upload + controls ----------
with st.sidebar:
    st.header("📁 Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = get_file_hash(file_bytes)

        # Only (re)process if this is a new/different file
        if st.session_state.active_file != file_hash:
            with st.spinner("Reading and indexing your document..."):
                tmp_dir = tempfile.mkdtemp()
                tmp_pdf_path = os.path.join(tmp_dir, uploaded_file.name)
                with open(tmp_pdf_path, "wb") as f:
                    f.write(file_bytes)

                try:
                    vectorstore = build_vectorstore(file_hash, tmp_pdf_path)
                    retriever = vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={
                            "k": 4,
                            "fetch_k": 10,
                            "lambda_mult": 0.5
                        }
                    )
                    st.session_state.retriever = retriever
                    st.session_state.active_file = file_hash
                    st.session_state.messages = []  # reset chat for new doc
                    st.success(f"'{uploaded_file.name}' indexed successfully!")
                except Exception as e:
                    st.error(f"Failed to process file: {e}")
                finally:
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    st.divider()
    st.header("About")
    st.write("Upload a PDF, then ask questions about it. Answers are grounded strictly in its content.")

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------- Main UI ----------
st.title("📄 RAG Chatbot")

if st.session_state.retriever is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    st.caption("Ask questions about your uploaded document below.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask a question about your document...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = get_llm()
                    prompt = get_prompt()
                    answer = get_answer(st.session_state.retriever, llm, prompt, query)
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})