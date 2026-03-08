# VaultAI Multimodal — RAG Intelligent pour PDFs & Images

Assistant RAG multimodal de nouvelle génération qui permet d'interroger simultanément vos **documents PDF** et vos **images** via une interface Premium.

**Pipeline** : Les PDFs sont découpés et vectorisés ; les images sont analysées par **LLaMA 3.2 Vision** et leurs descriptions sont indexées dans le même espace vectoriel. L'IA répond en s'appuyant strictement sur vos sources.

![Status](https://img.shields.io/badge/Status-Stable-success)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![LLM](https://img.shields.io/badge/LLM-Llama--3.1%20%2B%20Llama--3.2--Vision-orange)

## Fonctionnalités

- **Multimodal** : Indexation simultanée de PDFs (texte) et d'images (via Vision IA).
- **LLaMA 3.2 Vision** : Analyse automatique des images avec description détaillée.
- **Persistance** : Stockage vectoriel local via ChromaDB (`./chroma_db_multimodal`).
- **Interface Premium** : Design Glassmorphism avec bulles de chat animées.
- **Réglages IA** : Température, Top-K et paramètres de chunking ajustables.
- **Sources Citées** : Chaque réponse indique les documents/images consultés.

## Schéma Global du Pipeline

```text
┌──────────────────────────────────────────────────────────────┐
│                    ENTRÉES UTILISATEUR                        │
│                                                              │
│   📄 PDF              🖼️ Image (PNG/JPG)                     │
│     │                      │                                 │
│     ▼                      ▼                                 │
│  PyPDF2              LLaMA 3.2 Vision                        │
│  (Extraction          (Description                           │
│   de texte)            automatique)                          │
│     │                      │                                 │
│     ▼                      ▼                                 │
│  ┌─────────────────────────────────────┐                     │
│  │         TEXTE UNIFIÉ                │                     │
│  │  (texte PDF + descriptions images)  │                     │
│  └─────────────────────────────────────┘                     │
│                    │                                         │
│                    ▼                                         │
│           ┌────────────────┐                                 │
│           │   CHUNKING     │                                 │
│           │  (Découpage)   │                                 │
│           └────────────────┘                                 │
│                    │                                         │
│                    ▼                                         │
│         ┌──────────────────┐                                 │
│         │   EMBEDDINGS     │                                 │
│         │ all-MiniLM-L6-v2 │                                 │
│         │ (384 dimensions) │                                 │
│         └──────────────────┘                                 │
│                    │                                         │
│                    ▼                                         │
│          ┌─────────────────┐                                 │
│          │   CHROMADB      │                                 │
│          │  (Stockage      │                                 │
│          │   Vectoriel     │                                 │
│          │   Persistant)   │                                 │
│          └─────────────────┘                                 │
│                    │                                         │
│         Question utilisateur                                 │
│                    │                                         │
│                    ▼                                         │
│          ┌─────────────────┐                                 │
│          │  RETRIEVAL      │                                 │
│          │  (Recherche de  │                                 │
│          │   similarité    │                                 │
│          │   cosinus)      │                                 │
│          └─────────────────┘                                 │
│                    │                                         │
│                    ▼                                         │
│          ┌─────────────────┐                                 │
│          │  LLaMA 3.1 8B   │                                 │
│          │  (Génération    │                                 │
│          │   de réponse)   │                                 │
│          └─────────────────┘                                 │
│                    │                                         │
│                    ▼                                         │
│             💬 RÉPONSE                                       │
│          (avec sources citées)                               │
└──────────────────────────────────────────────────────────────┘
```

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| **Frontend** | Streamlit |
| **Vector DB** | ChromaDB (persistant) |
| **LLM Texte** | LLaMA 3.1 8B (via Groq) |
| **LLM Vision** | LLaMA 3.2 11B Vision (via Groq) |
| **Embeddings** | `all-MiniLM-L6-v2` (Sentence Transformers) |
| **Parsing PDF** | PyPDF2 |

## Installation

1. **Cloner le projet** :
   ```bash
   git clone https://github.com/wassima-azzouzi/rag_multimodal.git
   cd rag_multimodal
   ```

2. **Créer un environnement virtuel** :
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration** :
   Créez un fichier `.env` à la racine :
   ```env
   GROQ_API_KEY=votre_cle_groq
   ```

## Lancement
```bash
streamlit run app.py
```

## Utilisation
1. Chargez vos fichiers **PDF** et/ou **images** (PNG, JPG) dans la barre latérale.
2. Réglez les paramètres de chunking et d'IA si besoin.
3. Cliquez sur **Indexer** pour construire la base multimodale.
4. Posez vos questions — l'IA consultera textes ET descriptions d'images !

## Dépannage (Troubleshooting)

### Erreur API Groq : `Error code: 401 - Invalid API Key`
Cette erreur signifie que la clé API dans votre fichier `.env` est invalide, expirée, ou que Windows garde en mémoire une ancienne clé.

**Solutions :**

1. **Générez une nouvelle clé** depuis la [Console Groq](https://console.groq.com/keys) et mettez-la dans le fichier `.env`.
   
2. **Si l'erreur persiste sur Windows (PowerShell) :**
   Il se peut que le terminal garde en mémoire une ancienne clé. Pour forcer la nouvelle clé, exécutez ces commandes :
   ```powershell
   # 1. Fermez d'abord l'application en cours (Ctrl + C)
   
   # 2. Forcez la variable d'environnement avec votre NOUVELLE clé
   $env:GROQ_API_KEY="gsk_votre_nouvelle_cle_ici"
   
   # 3. Relancez l'application
   streamlit run app.py
   ```
   *Alternative* : Ouvrez simplement un tout nouveau terminal, réactivez l'environnement (`.\.venv\Scripts\activate`) et relancez `app.py`.
