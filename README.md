# RAG Multimodal – Application Streamlit

Ce projet est une application **RAG (Retrieval-Augmented Generation)** simple qui permet
d'uploader vos documents (PDF, TXT, DOCX/PPTX convertis en texte brut) et de poser des
questions dessus via une interface type chat.

## 1. Installation

```bash
git clone <ce_projet> rag_multimodal_app
cd rag_multimodal_app

# Créer un environnement virtuel (optionnel mais recommandé)
python -m venv .venv
source .venv/bin/activate  # sous Windows: .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

## 2. Configuration de la clé GROQ

Créer un fichier `.env` à la racine du projet :

```bash
cp .env.example .env
```

Éditer le fichier `.env` et ajouter votre clé :

```env
GROQ_API_KEY=your_groq_api_key_here
```

Vous pouvez obtenir une clé sur le site de Groq.

## 3. Lancer l'application

```bash
streamlit run app.py
```

## 4. Utilisation

1. Ouvrez l'URL locale indiquée par Streamlit (en général http://localhost:8501).
2. Dans la barre latérale :
   - Uploadez vos documents (PDF, TXT, etc.).
   - Ajustez la taille des chunks et le chevauchement si besoin.
   - Cliquez sur **"Construire la base RAG"**.
3. Dans la zone de chat en bas, posez vos questions sur le contenu des documents.
4. L'assistant répondra en se basant **uniquement** sur les documents indexés et affichera
   en plus les passages utilisés comme contexte.

## 5. Multimodalité ?

Ici, "multimodal" signifie principalement **multi-formats de documents texte**
(PDF, TXT, DOCX, PPTX, etc.).  
Pour ajouter le support d'images (questions sur des figures, captures, etc.), vous pouvez
étendre l'application en :

- ajoutant un modèle de vision (CLIP, vision encoder, etc.),
- extrayant des features d'images,
- et en les stockant dans la même base vectorielle ou dans une base séparée.

Cela dépasse le cadre du squelette de base fourni, mais l'architecture est prête à être
étendue.

---

Projet prêt à être modifié et adapté à vos besoins pédagogiques ou de projet.
