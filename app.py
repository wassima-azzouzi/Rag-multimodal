import streamlit as st
import os
import tempfile
import base64
import requests
from PIL import Image

# LangChain
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Groq pour le RAG textuel
from langchain_groq import ChatGroq


# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
st.set_page_config(page_title="RAG Multimodal", layout="wide")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -----------------------------------------------------------
# 1) LLM Groq pour le texte
# -----------------------------------------------------------
def build_llm():
    return ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )


# -----------------------------------------------------------
# 2) DeepSeek Vision pour description d’image
# -----------------------------------------------------------
def describe_image_with_deepseek(image_path):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return " ERREUR : DEEPSEEK_API_KEY non défini."

    # Lecture de l’image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Décris cette image en détail."},
                    {"type": "image", "image": f"data:image/png;base64,{img_b64}"}
                ]
            }
        ]
    }

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload
    )

    if response.status_code != 200:
        return f"Erreur API DeepSeek: {response.text}"

    return response.json()["choices"][0]["message"]["content"]


# -----------------------------------------------------------
# 3) Chargement documents
# -----------------------------------------------------------
def load_document(file):
    name = file.name.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=name) as tmp:
        tmp.write(file.read())
        tmp.flush()
        tmp_path = tmp.name

    if name.endswith(".pdf"):
        return PyPDFLoader(tmp_path).load()

    if name.endswith(".txt"):
        return TextLoader(tmp_path, encoding="utf-8").load()

    if name.endswith(".docx"):
        return Docx2txtLoader(tmp_path).load()

    if name.endswith(".pptx"):
        return UnstructuredPowerPointLoader(tmp_path).load()

    #  IMAGE → Description DeepSeek
    if name.endswith((".jpg", ".jpeg", ".png")):
        desc = describe_image_with_deepseek(tmp_path)
        text = f"[Description automatique de {file.name}]\n\n{desc}"
        return [Document(page_content=text, metadata={"source": file.name})]

    return []


# -----------------------------------------------------------
# 4) Création du vecteur-store
# -----------------------------------------------------------
def build_vectorstore(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=80
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    return vectordb, retriever


# -----------------------------------------------------------
# 5) Gestion chat
# -----------------------------------------------------------
def init_chat():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "retriever" not in st.session_state:
        st.session_state.retriever = None


def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})


def render_chat():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


# -----------------------------------------------------------
# 6) MAIN APP
# -----------------------------------------------------------
def main():
    st.title("Bienvenu RAG Multimodal ")
    st.write("PDF • TXT •  Images → Chat intelligent grâce au RAG multimodal.")

    init_chat()

    files = st.file_uploader(
        "Uploader vos fichiers",
        type=["pdf", "txt", "docx", "pptx", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button(" Construire la base RAG"):
        if not files:
            st.error("Veuillez ajouter des fichiers.")
            return

        docs = []
        for f in files:
            docs.extend(load_document(f))

        vectordb, retriever = build_vectorstore(docs)
        st.session_state.retriever = retriever

        st.success("Base RAG construite ! ")

    st.divider()

    render_chat()

    user_text = st.chat_input("Pose une question…")

    if user_text:
        add_message("user", user_text)

        retriever = st.session_state.retriever
        if retriever is None:
            st.error("⚠ Vous devez d'abord construire la base RAG.")
            return

        docs = retriever.invoke(user_text)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Tu es un assistant basé sur un système RAG.
Réponds uniquement à partir du CONTEXTE suivant :

CONTEXTE :
{context}

QUESTION :
{user_text}

RÉPONSE :
"""

        llm = build_llm()
        answer = llm.invoke(prompt).content

        add_message("assistant", answer)
        st.rerun()


if __name__ == "__main__":
    main()
