"""
rag.py  ·  DocuMind — Core RAG Engine
─────────────────────────────────────────────────────────────────────────────
DocuMind: Chat with your documents. Ask the world.

Routing logic:
  PDF mode     → question is about the uploaded document
  Web mode     → question needs real-time / post-2023 data (Tavily search)
  General mode → stable knowledge answered directly by the LLM
  DateTime     → current date/time answered locally via datetime.now()
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Generator

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────
CHROMA_DIR          = "./chroma_db"
EMBED_MODEL         = "all-MiniLM-L6-v2"        # runs locally, no API key
GROQ_MODEL          = "llama-3.3-70b-versatile"
CHUNK_SIZE          = 1000
CHUNK_OVERLAP       = 200
TOP_K               = 5
RELEVANCE_THRESHOLD = 0.35


# ── LLM factory ───────────────────────────────────────────────────────────────
def get_groq_llm(temperature: float = 0.3, streaming: bool = False) -> ChatGroq:
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        max_tokens=1024,
        streaming=streaming,
    )


# ── Embeddings (singleton) ────────────────────────────────────────────────────
try:
    from langchain_huggingface import HuggingFaceEmbeddings as _HFE
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings as _HFE  # type: ignore

_embeddings = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = _HFE(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _embeddings


# ── PDF loading & splitting ───────────────────────────────────────────────────
def load_and_split(pdf_path: str) -> list:
    """Load a PDF and return cleaned, page-tagged chunks."""
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
def build_vectorstore(chunks: list, collection_name: str = "default") -> Chroma:
    return Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_name=collection_name,
    )


def load_vectorstore(collection_name: str = "default") -> Chroma:
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_get_embeddings(),
        collection_name=collection_name,
    )


# ── RAG chain ─────────────────────────────────────────────────────────────────
_CONDENSE_PROMPT = PromptTemplate.from_template(
    """Given the conversation history and a follow-up question, rephrase the
follow-up into a standalone question.

Chat History:
{chat_history}

Follow-up: {question}
Standalone question:"""
)

_QA_PROMPT = PromptTemplate.from_template(
    """You are DocuMind, a precise AI assistant that answers questions about documents.

Use ONLY the context below. If the answer isn't in the context, say
"I couldn't find that in the document." — never fabricate.

Context:
{context}

Question: {question}

Answer (be concise; cite page numbers when relevant):"""
)


def build_rag_chain(vectorstore: Chroma) -> ConversationalRetrievalChain:
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20},
    )
    return ConversationalRetrievalChain.from_llm(
        llm=get_groq_llm(temperature=0.2),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=_CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
        verbose=False,
    )


# ── Intent patterns ───────────────────────────────────────────────────────────

# 1. PDF-directed questions — always routed to RAG regardless of similarity score
_PDF_INTENT = re.compile(
    r"\b("
    r"summari[sz]e|summary|summarise|"
    r"this pdf|the pdf|given pdf|uploaded pdf|"
    r"this document|the document|this doc|the doc|"
    r"according to|based on the|from the pdf|in the pdf|"
    r"what does (it|the pdf|the doc|this) say|"
    r"tell me about (the|this) (pdf|document|doc)|"
    r"explain (the|this) (pdf|document)|"
    r"key points|main points|overview|"
    r"what is (the|this) (document|pdf) about"
    r")\b",
    re.IGNORECASE,
)

# 2. Date/time — answered locally; never needs web search
_DATETIME = re.compile(
    r"\b("
    r"what('s| is)(\s+the)?(\s+today'?s?)?\s+(date|day|time|month|year)|"
    r"what is (the )?today|"
    r"today'?s?\s+date|"
    r"current\s+(date|time|day)|"
    r"what day is (it|today)|"
    r"tell me (the\s+)?(date|time|day)|"
    r"whats? (today|the date|the time|the day)"
    r")\b",
    re.IGNORECASE,
)

# 3. Real-time questions — route to Tavily web search
_REALTIME = re.compile(
    r"\b("
    r"yesterday|today|tonight|right now|just now|this morning|this evening|"
    r"this week|this month|this year|last night|last week|last month|"
    r"latest|recent|current|live|ongoing|happening|"
    r"just announced|breaking|just released|out now|"
    r"2024|2025|2026|"
    r"news|headlines?|top stor(?:y|ies)|what(?:'s| is) happening|"
    r"what(?:'s| is) going on|updates?|developments?|"
    r"did .{1,30} happen|has .{1,30} happened|"
    r"price of|stock price|share price|market cap|exchange rate|"
    r"how much (?:is|does|did)|worth today|"
    r"weather|temperature|forecast|will it rain|"
    r"who (?:is|are|won|leads?|became|got|was arrested|was appointed)|"
    r"who(?:'s| is) the (?:current|new|latest)|"
    r"score of|result of|winner of|match result|"
    r"championship|election|verdict|"
    r"what (?:is|are) the (?:current|latest|new|recent)|"
    r"when (?:is|was|did|does)|"
    r"ipl|cricket|football|soccer|nba|nfl|premier league|world cup|"
    r"olympics?|tournament|"
    r"release date|launch date|new (?:version|model|phone|update|feature)"
    r")\b",
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
def _is_pdf_question(vectorstore: Chroma, question: str) -> tuple[bool, list]:
    if _has_pdf_intent(question):
        return True, []
    try:
        results  = vectorstore.similarity_search_with_relevance_scores(question, k=TOP_K)
        top_docs = [doc for doc, score in results if score >= RELEVANCE_THRESHOLD]
        return len(top_docs) > 0, top_docs
    except Exception:
        return True, []


# ── Web search via Tavily ─────────────────────────────────────────────────────
def _web_search(query: str, max_results: int = 4) -> list[dict]:
    key = os.getenv("TAVILY_API_KEY")
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
    "You are DocuMind, a knowledgeable AI assistant with real-time web search.\n"
    "Use provided search results (if any) and prioritise them over training data.\n"
    "Cite sources by title when using web results. Be concise and honest.\n"
    "Current date and time: {dt}"
)


def _answer_general(
    question: str,
    chat_history: list[dict],
) -> tuple[str, list[dict], bool]:
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
            search_ctx += (
                f"[{i}] {r.get('title','')} ({r.get('url','')})\n"
                f"{r.get('content','')[:500]}\n\n"
            )

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
def ask(
    chain: ConversationalRetrievalChain | None,
    question: str,
    chat_history: list[dict] | None = None,
    vectorstore: Chroma | None = None,
) -> dict:
    """
    Route the question and return:
      answer, sources, snippets, mode, web_sources
    mode ∈ { "pdf", "web", "web_no_key", "web_failed", "general" }
    """
    chat_history = chat_history or []

    is_pdf_q = False
    if chain is not None and vectorstore is not None:
        is_pdf_q, _ = _is_pdf_question(vectorstore, question)

    if not is_pdf_q:
        answer, web_results, web_attempted = _answer_general(question, chat_history)
        used_web = bool(web_results)

        if used_web:
            mode = "web"
        elif web_attempted and not os.getenv("TAVILY_API_KEY"):
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

    # PDF RAG
    result   = chain.invoke({"question": question})
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
        "You are DocuMind. Read this PDF excerpt and produce a concise summary "
        "(5–8 bullet points) covering: main topic, key findings, and conclusions.\n\n"
        f"Excerpt:\n{sample}\n\nSummary:"
    )
    return llm.invoke(prompt).content.strip()