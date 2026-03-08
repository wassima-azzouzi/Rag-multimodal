import os
import tempfile
import streamlit as st
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import PyPDF2
from openai import OpenAI
from dotenv import load_dotenv
import base64
from PIL import Image
import io

# Charger les variables d'environnement (.env)
load_dotenv()

# -------------------------------------------------------------
# CONFIGURATION & STYLE CSS PREMIUM (Glassmorphism)
# -------------------------------------------------------------
st.set_page_config(page_title="VaultAI Multimodal - RAG Premium", layout="wide", page_icon="🖼️")

def local_css():
    st.markdown("""
    <style>
    /* Global Background */
    .stApp {
        background: radial-gradient(circle at top right, #0f172a, #1e293b);
        color: #f1f5f9;
    }
    
    /* Input Area */
    .stChatInputContainer {
        padding-bottom: 2rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Custom Chat Bubbles */
    .chat-bubble {
        padding: 1.2rem 1.5rem;
        border-radius: 18px;
        margin-bottom: 1.2rem;
        display: flex;
        flex-direction: column;
        animation: fadeIn 0.4s ease-out;
        max-width: 80%;
        line-height: 1.5;
        font-family: 'Inter', sans-serif;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        align-self: flex-end;
        color: white;
        border-bottom-right-radius: 4px;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }
    
    .assistant-bubble {
        background: rgba(30, 41, 59, 0.7);
        align-self: flex-start;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-bottom-left-radius: 4px;
        color: #e2e8f0;
        backdrop-filter: blur(5px);
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Titles */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px -10px rgba(59, 130, 246, 0.5);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# -------------------------------------------------------------
# LOGIQUE CHROMADB & RAG MULTIMODAL
# -------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

# Initialisation ChromaDB (persistant)
CHROMA_PATH = "./chroma_db_multimodal"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Collection
collection = chroma_client.get_or_create_collection(
    name="multimodal_docs", 
    embedding_function=embedding_function
)

def encode_image(image_path):
    """Encode une image en base64 pour l'API Vision."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def describe_image(image_path):
    """Utilise LLaMA 3.2 Vision via Groq pour décrire une image."""
    base64_image = encode_image(image_path)
    
    try:
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Décris cette image en détail pour un système RAG. Sois précis sur les objets, le texte visible, les graphiques et le contexte général."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur Vision : {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extraire le texte d'un fichier PDF."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def chunk_text(text, size=800, overlap=100):
    """Découper le texte en morceaux avec chevauchement."""
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks

def get_answer(question, chat_history, top_k=4, temperature=0.7):
    """Générer une réponse en utilisant le pipeline RAG."""
    # 1. Recherche de similarité dans ChromaDB
    results = collection.query(query_texts=[question], n_results=top_k)
    context = "\n\n".join(results['documents'][0])
    
    # 2. Construction du prompt avec instructions strictes
    messages = [
        {"role": "system", "content": f"""Tu es VaultAI Multimodal, un assistant RAG expert. 
        Tes instructions sont STRICTES :
        1. Réponds UNIQUEMENT en utilisant le contexte fourni ci-dessous.
           Le contexte peut contenir du texte extrait de PDFs ET des descriptions d'images.
        2. Si la réponse n'est pas dans le contexte, dis explicitement :
           "Désolé, je ne trouve pas cette information dans vos documents."
        3. Ne fais pas appel à tes connaissances générales.
        4. Cite le nom du document ou de l'image source si possible.
        """}
    ]
    
    # Ajouter l'historique récent
    for msg in chat_history[-5:]:
        messages.append(msg)
        
    messages.append({
        "role": "user", 
        "content": f"CONTEXTE:\n{context}\n\nQUESTION: {question}"
    })
    
    try:
        # 3. Génération via LLM
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=temperature
        )
        return resp.choices[0].message.content, results['metadatas'][0]
    except Exception as e:
        return f"Erreur API Groq : {e}", []

# -------------------------------------------------------------
# INTERFACE STREAMLIT PREMIUM
# -------------------------------------------------------------
st.title("🖼️ VaultAI - RAG Multimodal")

# Sidebar
with st.sidebar:
    st.header("⚙️ Réglages & Paramètres")
    
    st.subheader("🤖 Intelligence")
    temp = st.slider(
        label="Niveau de créativité (Température)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.7, 
        step=0.1,
        help="0.0 = Factuel | 1.0 = Créatif"
    )
    
    top_k = st.slider(
        label="Profondeur de recherche (Top-K)", 
        min_value=1, 
        max_value=10, 
        value=4,
        help="Nombre de morceaux lus par l'IA."
    )
    
    st.divider()
    st.subheader("📝 Découpage (Chunking)")
    chunk_size = st.number_input(
        label="Taille des morceaux", 
        min_value=100, 
        max_value=2000, 
        value=800
    )
    
    chunk_overlap = st.number_input(
        label="Chevauchement (Overlap)", 
        min_value=0, 
        max_value=500, 
        value=100
    )

    st.divider()
    st.header("📂 Documents & Images")
    uploaded_files = st.file_uploader(
        "Indexez vos PDFs et Images", 
        type=["pdf", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("✨ Indexer")
    with col2:
        clear_btn = st.button("🗑️ Vider")

    if build_btn:
        if uploaded_files:
            with st.status("🛠️ Pipeline RAG Multimodal en cours...", expanded=True) as status:
                for f in uploaded_files:
                    # --- PDF ---
                    if f.type == "application/pdf":
                        st.write(f"📄 **PDF** : {f.name}")
                        st.write("   1. Extraction du texte...")
                        text = extract_text_from_pdf(f)
                        
                        st.write("   2. Création des chunks...")
                        chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
                        
                        st.write("   3. Vectorisation & Stockage ChromaDB...")
                        ids = [f"{f.name}_{i}" for i in range(len(chunks))]
                        metadatas = [{"source": f.name, "type": "pdf"} for _ in chunks]
                        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
                        st.write(f"   ✅ {len(chunks)} chunks indexés.")
                    
                    # --- IMAGE ---
                    elif f.type in ["image/png", "image/jpeg"]:
                        st.write(f"🖼️ **Image** : {f.name}")
                        st.write("   1. Envoi à LLaMA 3.2 Vision...")
                        
                        # Sauvegarder temporairement l'image
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                            tmp.write(f.read())
                            tmp_path = tmp.name
                        
                        description = describe_image(tmp_path)
                        st.write(f"   2. Description générée ({len(description)} caractères)")
                        
                        st.write("   3. Vectorisation de la description...")
                        collection.add(
                            ids=[f"img_{f.name}"],
                            documents=[f"[Image: {f.name}] Description automatique : {description}"],
                            metadatas=[{"source": f.name, "type": "image"}]
                        )
                        os.unlink(tmp_path)
                        st.write("   ✅ Image indexée.")
                
                status.update(label="✔ Base Multimodale Construite !", state="complete", expanded=False)
                st.success(f"{len(uploaded_files)} ressource(s) indexée(s).")
        else:
            st.error("Veuillez uploader au moins un fichier.")
            
    if clear_btn:
        ids = collection.get()['ids']
        if ids:
            collection.delete(ids=ids)
            st.success("Base de données vidée.")
        else:
            st.info("La base est déjà vide.")

    # Aperçu de la base
    st.divider()
    st.header("📊 Dossier Vectoriel")
    count = collection.count()
    st.metric("Éléments en base", count)
    if count > 0:
        with st.expander("Sources indexées"):
            all_meta = collection.get()['metadatas']
            sources = list(set([m['source'] for m in all_meta]))
            for s in sources:
                file_type = next((m['type'] for m in all_meta if m['source'] == s), "unknown")
                icon = "📄" if file_type == "pdf" else "🖼️"
                st.write(f"{icon} {s}")

    st.divider()
    st.write("Statut: **Prêt**" if GROQ_API_KEY else "Statut: ⚠️ **Clé API manquante**")

# Chat Interface
if "messages_multimodal" not in st.session_state:
    st.session_state.messages_multimodal = []

for msg in st.session_state.messages_multimodal:
    role_class = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f'<div class="chat-bubble {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)

user_input = st.chat_input("Posez une question sur vos documents ou vos images...")

if user_input:
    st.markdown(f'<div class="chat-bubble user-bubble">{user_input}</div>', unsafe_allow_html=True)
    
    with st.spinner("🔍 Recherche sémantique & Génération..."):
        answer, sources = get_answer(user_input, st.session_state.messages_multimodal, top_k=top_k, temperature=temp)
        
        st.session_state.messages_multimodal.append({"role": "user", "content": user_input})
        st.session_state.messages_multimodal.append({"role": "assistant", "content": answer})
        
        st.markdown(f'<div class="chat-bubble assistant-bubble">{answer}</div>', unsafe_allow_html=True)
        
        if sources:
            source_names = list(set([s['source'] for s in sources]))
            st.caption(f"📎 Sources : {', '.join(source_names)}")
            
    st.rerun()
