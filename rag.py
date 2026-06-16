"""
rag.py  ·  DocuMind — Core RAG Engine
Uses modern LangChain LCEL (no ConversationalRetrievalChain) — Python 3.14 safe.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()


# ── Secret resolution (local .env  →  Streamlit Cloud secrets) ────────────────
def _get_secret(key: str) -> str | None:
    """Read a secret from the environment first, then from st.secrets.
    Works both locally (via .env / shell env) and on Streamlit Cloud.
    """
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_DIR          = "./chroma_db"
GROQ_MODEL          = "llama-3.3-70b-versatile"
CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 200
TOP_K               = 5
RELEVANCE_THRESHOLD = 0.35

try:
    EMBED_MODEL = "all-MiniLM-L6-v2"
    from langchain_huggingface import HuggingFaceEmbeddings as _HFE
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings as _HFE


# ── LLM factory ───────────────────────────────────────────────────────────────
def get_groq_llm(temperature: float = 0.3) -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=_get_secret("GROQ_API_KEY"),
        temperature=temperature,
        max_tokens=1024,
    )


# ── Embeddings (singleton) ────────────────────────────────────────────────────
_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = _HFE(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


# ── PDF loading ───────────────────────────────────────────────────────────────
def load_and_split(pdf_path: str) -> list:
    loader = PyPDFLoader(pdf_path)
    pages  = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for chunk in chunks:
        chunk.page_content = re.sub(r"\s+", " ", chunk.page_content).strip()
    return [c for c in chunks if len(c.page_content) > 50]


# ── Vector store ──────────────────────────────────────────────────────────────
def build_vectorstore(chunks: list, collection_name: str = "default"):
    import shutil
    store_dir = os.path.join(CHROMA_DIR, collection_name)
    if os.path.exists(store_dir):
        shutil.rmtree(store_dir)
    return Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=store_dir,
    )


def load_vectorstore(collection_name: str = "default"):
    store_dir = os.path.join(CHROMA_DIR, collection_name)
    return Chroma(
        persist_directory=store_dir,
        embedding_function=_get_embeddings(),
    )


# ── LCEL RAG chain (replaces ConversationalRetrievalChain) ────────────────────
_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are DocuMind, a precise assistant that answers questions about documents.\n"
     "Use ONLY the context below. If the answer is not in the context, say "
     "'I couldn't find that in the document.' Never fabricate.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

_CONDENSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Given the conversation history and a follow-up question, "
     "rephrase it as a standalone question. Return ONLY the question."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])


def build_rag_chain(vectorstore):
    """Returns a simple callable: fn(question, chat_history) -> dict"""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20},
    )
    llm = get_groq_llm(temperature=0.2)

    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def run(question: str, chat_history: list = None):
        chat_history = chat_history or []

        # Convert history to LangChain messages
        messages = []
        for msg in chat_history[-6:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # Condense follow-up into standalone question (only if history exists)
        standalone = question
        if messages:
            condense_chain = _CONDENSE_PROMPT | llm | StrOutputParser()
            standalone = condense_chain.invoke({
                "chat_history": messages,
                "question": question,
            })

        # Retrieve docs
        docs = retriever.invoke(standalone)

        # Generate answer
        qa_chain = _QA_PROMPT | llm | StrOutputParser()
        answer = qa_chain.invoke({
            "context":      _format_docs(docs),
            "chat_history": messages,
            "question":     question,
        })

        return {"answer": answer, "source_documents": docs}

    # Attach vectorstore reference for relevance check
    run.vectorstore = vectorstore
    return run


# ── Intent patterns ───────────────────────────────────────────────────────────
_PDF_INTENT = re.compile(
    r"\b(summari[sz]e|summary|summarise|"
    r"this pdf|the pdf|given pdf|uploaded pdf|"
    r"this document|the document|this doc|the doc|"
    r"according to|based on the|from the pdf|in the pdf|"
    r"key points|main points|overview|"
    r"what is (the|this) (document|pdf) about)\b",
    re.IGNORECASE,
)

_DATETIME = re.compile(
    r"\b(what('s| is)(\s+the)?(\s+today'?s?)?\s+(date|day|time|month|year)|"
    r"what is (the )?today|today'?s?\s+date|"
    r"current\s+(date|time|day)|what day is (it|today)|"
    r"tell me (the\s+)?(date|time|day)|"
    r"whats? (today|the date|the time|the day))\b",
    re.IGNORECASE,
)

_REALTIME = re.compile(
    r"\b(yesterday|today|tonight|right now|just now|this morning|this evening|"
    r"this week|this month|this year|last night|last week|last month|"
    r"latest|recent|current|live|ongoing|happening|"
    r"just announced|breaking|just released|out now|2024|2025|2026|"
    r"news|headlines?|top stor(?:y|ies)|what(?:'s| is) happening|"
    r"what(?:'s| is) going on|updates?|developments?|"
    r"price of|stock price|share price|market cap|exchange rate|"
    r"how much (?:is|does|did)|worth today|"
    r"weather|temperature|forecast|will it rain|"
    r"who (?:is|are|won|leads?|became|got|was arrested|was appointed)|"
    r"who(?:'s| is) the (?:current|new|latest)|"
    r"score of|result of|winner of|match result|"
    r"championship|election|verdict|"
    r"ipl|cricket|football|soccer|nba|nfl|premier league|world cup|"
    r"olympics?|tournament|"
    r"release date|launch date|new (?:version|model|phone|update|feature))\b",
    re.IGNORECASE,
)


def _has_pdf_intent(q: str) -> bool:
    return bool(_PDF_INTENT.search(q))

def _is_datetime_query(q: str) -> bool:
    return bool(_DATETIME.search(q))

def _needs_web_search(q: str) -> bool:
    if _is_datetime_query(q):
        return False
    return bool(_REALTIME.search(q))


# ── Relevance check ───────────────────────────────────────────────────────────
def _is_pdf_question(vectorstore, question: str) -> tuple:
    if _has_pdf_intent(question):
        return True, []
    try:
        results  = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
        top_docs = [doc for doc, score in results if score >= RELEVANCE_THRESHOLD]
        return len(top_docs) > 0, top_docs
    except Exception:
        return True, []


# ── Web search ────────────────────────────────────────────────────────────────
def _web_search(query: str, max_results: int = 4) -> list:
    key = _get_secret("TAVILY_API_KEY")
    if not key:
        return []
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        return client.search(query=query, search_depth="basic",
                             max_results=max_results).get("results", [])
    except Exception as e:
        print(f"[DocuMind/Tavily] {e}")
        return []


# ── General / web answer ──────────────────────────────────────────────────────
_SYSTEM = (
    "You are DocuMind, a helpful AI assistant.\n"
    "Use provided search results (if any) and prioritise them over training data.\n"
    "Be concise and honest. Current date and time: {dt}"
)

def _answer_general(question: str, chat_history: list) -> tuple:
    llm           = get_groq_llm(temperature=0.4)
    web_results   = []
    web_attempted = False

    if _needs_web_search(question):
        web_attempted = True
        web_results   = _web_search(question)

    search_ctx = ""
    if web_results:
        search_ctx = "### Live Web Results\n"
        for i, r in enumerate(web_results, 1):
            search_ctx += f"[{i}] {r.get('title','')} ({r.get('url','')})\n{r.get('content','')[:500]}\n\n"

    history = ""
    for t in chat_history[-6:]:
        role     = "User" if t["role"] == "user" else "Assistant"
        history += f"{role}: {t['content']}\n"

    dt     = datetime.now().strftime("%A, %d %B %Y, %I:%M %p")
    system = _SYSTEM.format(dt=dt)
    prompt = (
        f"{system}\n\n"
        + (f"{search_ctx}\n" if search_ctx else "")
        + (f"Conversation:\n{history}\n" if history else "")
        + f"User: {question}\nDocuMind:"
    )

    resp = llm.invoke(prompt)
    return resp.content.strip(), web_results, web_attempted


# ── Primary ask ───────────────────────────────────────────────────────────────
def ask(chain, question: str, chat_history: list = None, vectorstore=None) -> dict:
    chat_history = chat_history or []

    is_pdf_q = False
    if chain is not None and vectorstore is not None:
        is_pdf_q, _ = _is_pdf_question(vectorstore, question)

    if not is_pdf_q:
        answer, web_results, web_attempted = _answer_general(question, chat_history)
        used_web = bool(web_results)

        if used_web:
            mode = "web"
        elif web_attempted and not _get_secret("TAVILY_API_KEY"):
            mode = "web_no_key"
        elif web_attempted:
            mode = "web_failed"
        else:
            mode = "general"

        return {
            "answer":      answer,
            "sources":     [r.get("url", "") for r in web_results],
            "snippets":    [r.get("content", "")[:220] + "…" for r in web_results],
            "mode":        mode,
            "web_sources": web_results,
        }

    # PDF RAG via LCEL chain
    result   = chain(question, chat_history)
    sources, snippets, seen = [], [], set()

    for doc in result.get("source_documents", []):
        page = (doc.metadata.get("page") or 0) + 1
        src  = f"Page {page}"
        if src not in seen:
            seen.add(src)
            sources.append(src)
            snip = doc.page_content[:220].strip()
            snippets.append(snip + ("…" if len(doc.page_content) > 220 else ""))

    return {
        "answer":      result["answer"],
        "sources":     sources,
        "snippets":    snippets,
        "mode":        "pdf",
        "web_sources": [],
    }


# ── PDF summariser ────────────────────────────────────────────────────────────
def summarise_pdf(chunks: list) -> str:
    sample = ""
    for chunk in chunks:
        if len(sample) > 3000:
            break
        sample += chunk.page_content + "\n\n"

    llm    = get_groq_llm(temperature=0.3)
    prompt = (
        "You are DocuMind. Produce a concise summary (5–8 bullet points) "
        "covering: main topic, key findings, and conclusions.\n\n"
        f"Excerpt:\n{sample}\n\nSummary:"
    )
    return llm.invoke(prompt).content.strip()