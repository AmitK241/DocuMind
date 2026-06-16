<div align="center">

# 🧠 DocuMind

### *Chat with your documents. Ask the world. Ask anything.*

**AI-powered chatbot that chats with PDFs, searches the web in real-time, and answers general questions**
**— built with Groq, LangChain, ChromaDB & Streamlit.**

<br>

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://iwixvgu3xkvcakgxtwcyyp.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-AmitK241-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AmitK241/DocuMind)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)

<br>

![DocuMind Banner]([https://opengraph.githubassets.com/2567c3c8eb1935c0b87cf94cd7456e0e4175e20a067292a31fe866ae78cabdf3/AmitK241/DocuMind](https://github.com/AmitK241/DocuMind/blob/main/Screenshot%202026-06-16%20220404.png))

</div>

---

## ✨ What is DocuMind?

**DocuMind** is a full-stack AI assistant with **three modes of intelligence**:

| Mode | Trigger | What it does |
|------|---------|--------------|
| 📄 **PDF RAG** | Upload a PDF & ask about it | Retrieves relevant chunks from your document using vector similarity search and answers with source citations |
| 🔍 **Live Web Search** | Ask about news, prices, events, current affairs | Detects real-time intent and searches the web via Tavily, then synthesizes an answer |
| 🤖 **General AI** | Ask anything else | Falls back to Groq's Llama 3.3 70B for fast, smart general answers |

No hallucinations on documents — if it's not in the PDF, DocuMind says so.

---

## 🎯 Key Features

- **🧠 Smart Intent Detection** — Automatically routes each question to the right mode (PDF / Web / General) using regex pattern matching, no manual mode switching needed
- **📄 Multi-PDF Support** — Upload and switch between multiple PDFs in a single session, each with its own ChromaDB collection
- **🔍 Real-Time Web Search** — Powered by [Tavily](https://tavily.com), with smart detection for news, prices, sports scores, weather, and breaking events
- **💬 Conversational Memory** — Maintains full multi-turn chat history; uses a condense-question step to handle follow-up questions correctly
- **📋 PDF Summarization** — Auto-generates a concise 5–8 bullet point summary when a PDF is uploaded
- **📌 Source Citations** — PDF answers show exact page references; web answers show clickable source cards with snippets
- **💾 Chat Export** — Download your entire chat session as a `.txt` file
- **🎨 Dark Glassmorphism UI** — Pure black background with blue/purple/green neon accents, Inter + JetBrains Mono fonts
- **☁️ Streamlit Cloud Ready** — Handles pysqlite3 monkey-patch for ChromaDB compatibility on Streamlit Cloud

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────┐
│           Intent Detection              │
│   (regex patterns: PDF / Web / General) │
└─────────────────────────────────────────┘
         │               │               │
         ▼               ▼               ▼
  ┌────────────┐  ┌────────────┐  ┌────────────┐
  │  PDF RAG   │  │ Web Search │  │ General LLM│
  │            │  │  (Tavily)  │  │            │
  │ ChromaDB   │  │            │  │ Groq LLaMA │
  │ (MMR ret.) │  │ Top 4 URLs │  │ 3.3 70B    │
  │ Top-K=5    │  │ + snippets │  │            │
  └────────────┘  └────────────┘  └────────────┘
         │               │               │
         ▼               ▼               ▼
┌─────────────────────────────────────────┐
│     Groq LLaMA 3.3 70B (Generator)      │
│   + Chat History (last 6 turns)         │
└─────────────────────────────────────────┘
         │
         ▼
   Structured Response
   { answer, sources, snippets, mode }
```

**Core files:**

```
DocuMind/
├── app.py          # Streamlit frontend — UI, session state, chat rendering
├── rag.py          # Core RAG engine — PDF loading, vector store, intent detection, web search
├── requirements.txt
└── chroma_db/      # Persisted vector collections (gitignored)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq API — `llama-3.3-70b-versatile` |
| **RAG Framework** | LangChain LCEL (Runnable chains, no deprecated `ConversationalRetrievalChain`) |
| **Vector Store** | ChromaDB with MMR retrieval (`k=5`, `fetch_k=20`) |
| **Embeddings** | `all-MiniLM-L6-v2` via HuggingFace (local, CPU) |
| **PDF Parsing** | PyPDFLoader + RecursiveCharacterTextSplitter (`chunk_size=1000`, `overlap=200`) |
| **Web Search** | Tavily Python Client |
| **Frontend** | Streamlit with custom CSS (glassmorphism, Inter font, dark theme) |
| **Deployment** | Streamlit Cloud |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [Groq API Key](https://console.groq.com) (free tier available)
- [Tavily API Key](https://app.tavily.com) (free tier — for web search)

### 1. Clone the repo

```bash
git clone https://github.com/AmitK241/DocuMind.git
cd DocuMind
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here   # Optional — web search won't work without it
```

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## ☁️ Deploying to Streamlit Cloud

1. Push your code to GitHub (make sure `chroma_db/` is in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. In **App Settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key"
TAVILY_API_KEY = "your_tavily_api_key"
```

> **Note:** DocuMind automatically handles the `pysqlite3-binary` monkey-patch required for ChromaDB to work on Streamlit Cloud's older SQLite environment.

---

## 🧪 How It Works (Under the Hood)

### PDF RAG Pipeline

1. **Load** → `PyPDFLoader` extracts text page by page
2. **Split** → `RecursiveCharacterTextSplitter` creates ~1000 char chunks with 200 char overlap
3. **Embed** → `all-MiniLM-L6-v2` encodes chunks into vectors
4. **Store** → Persisted in ChromaDB (separate collection per PDF)
5. **Retrieve** → MMR search finds top-5 relevant chunks
6. **Condense** → Follow-up questions are rephrased into standalone questions
7. **Generate** → Groq Llama answers with strict context grounding

### Intent Detection

DocuMind uses regex pattern matching to decide routing:

```python
# Detects PDF-specific questions ("summarize this doc", "according to the PDF")
_PDF_INTENT = re.compile(r"\b(summari[sz]e|this pdf|the document|...)\b")

# Detects real-time queries (news, prices, sports, weather)
_REALTIME = re.compile(r"\b(latest|breaking|price of|who won|weather|...)\b")
```

Relevance threshold (`0.35`) is also used as a secondary PDF-routing signal via ChromaDB similarity scores.

---

## 📸 Demo

> **Try it live:** [https://iwixvgu3xkvcakgxtwcyyp.streamlit.app/](https://iwixvgu3xkvcakgxtwcyyp.streamlit.app/)

**Example queries to try:**
- Upload any research paper → *"Summarize the key findings"*
- *"What's the latest news about AI today?"*
- *"What is the current price of Bitcoin?"*
- *"Who won the IPL 2025?"*
- Upload a resume → *"What are the candidate's top skills?"*

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ Yes | Powers the LLM (Llama 3.3 70B). Get free at [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | ⚡ Optional | Enables live web search. Get free at [app.tavily.com](https://app.tavily.com). Without it, web queries fall back to general AI. |

---

## 📦 Dependencies

```
langchain >= 0.3.26
langchain-groq >= 0.3.2
langchain-huggingface >= 0.1.2
chromadb == 0.5.23
pysqlite3-binary          # Streamlit Cloud SQLite fix
sentence-transformers >= 3.4.1
pypdf >= 5.1.0
streamlit >= 1.41.1
tavily-python >= 0.5.0
python-dotenv >= 1.0.1
```

---

## 🗺️ Roadmap

- [ ] Multi-document cross-PDF Q&A
- [ ] Image/chart extraction from PDFs (multimodal RAG)
- [ ] User authentication + saved chat history
- [ ] Support for DOCX, TXT, and CSV files
- [ ] Agent-based mode with tool use
- [ ] Streaming responses for real-time token output

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Amit Kumar**
B.Tech CSE | MMNIT Allahabad, Prayagraj

[![GitHub](https://img.shields.io/badge/GitHub-AmitK241-181717?style=flat-square&logo=github)](https://github.com/AmitK241)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**If you found this useful, drop a ⭐ — it means a lot!**

*Built with 🧠 + ☕ using Groq, LangChain, and Streamlit*

</div>
