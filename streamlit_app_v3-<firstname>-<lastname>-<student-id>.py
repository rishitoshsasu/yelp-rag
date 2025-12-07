import glob
import os
from pathlib import Path

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings

# Retrieval
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory

# Model + history
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from prompt import SYSTEM_PROMPT

# ------------------------
# BASIC CONFIG
# ------------------------
st.set_page_config(page_title="Yelp Bot (Minimal RAG)", page_icon="🤖")
st.title("Yelp Bot")

DATA_DIR = Path("data")
MODEL_NAME = "gemini-2.5-flash"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, CPU-friendly

# API key (either Streamlit secrets or environment)
API_KEY = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    st.error(
        "Missing GEMINI_API_KEY. Add it in Streamlit secrets or as an environment variable."
    )
    st.stop()

# Ensure session defaults exist before sidebar uses them
if "_active_settings" not in st.session_state:
    st.session_state["_active_settings"] = {
        "chunk_size": 1200,
        "chunk_overlap": 150,
        "k": 4,
        "temperature": 0.2,
    }


# ------------------------
# LIGHT HELPERS
# ------------------------
def _load_md_documents(folder: Path):
    docs = []
    for path in sorted(glob.glob(str(folder / "**/*.md"), recursive=True)):
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        docs.append(Document(page_content=text, metadata={"source": path}))
    return docs


def _split_docs(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)


def _settings_key(chunk_size, chunk_overlap):
    # must be hashable for Streamlit cache
    return (
        chunk_size,
        chunk_overlap,
    )


@st.cache_resource(show_spinner=True)
def _build_vectorstore_cached(settings_key):
    # read current settings from session_state (set by sidebar)
    s = st.session_state.get("_active_settings", {})
    docs = _load_md_documents(DATA_DIR)
    if not docs:
        raise RuntimeError(f"No .md files found in {DATA_DIR}/ — add at least one.")

    # TODO 1: using _split_docs, split the documents
    chunks = 
    # TODO 2: create HuggingFaceEmbeddings with EMBED_MODEL, also normalize the embeddings
    # reference: https://api.python.langchain.com/en/latest/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html
    embeddings = 
    return FAISS.from_documents(chunks, embeddings)


def _clear_index_cache():
    _build_vectorstore_cached.clear()


def _render_bubble(role: str, text: str):
    # Use Streamlit's native chat messages so Markdown renders properly
    chat_role = "assistant" if role == "assistant" else "user"
    with st.chat_message(chat_role):
        st.markdown(text)


# ------------------------
# SIDEBAR: MINIMAL CONTROLS
# ------------------------
with st.sidebar:
    # --- Chunking (document splitter) settings ---
    st.subheader("Chunking Settings")
    with st.form("chunking_form"):
        st.slider(
            "Chunk size",
            min_value=200,
            max_value=3000,
            value=st.session_state["_active_settings"]["chunk_size"],
            step=50,
            key="ui_chunk_size",
        )
        st.slider(
            "Chunk overlap",
            min_value=0,
            max_value=800,
            value=st.session_state["_active_settings"]["chunk_overlap"],
            step=10,
            key="ui_chunk_overlap",
        )
        chunk_submitted = st.form_submit_button("Apply & Rebuild Index")
    if chunk_submitted:
        st.session_state["_active_settings"].update(
            {
                "chunk_size": st.session_state["ui_chunk_size"],
                "chunk_overlap": st.session_state["ui_chunk_overlap"],
            }
        )
        _clear_index_cache()
        # Reset chat history so a fresh conversation starts after rebuild
        try:
            del st.session_state["chat_history"]
        except KeyError:
            pass
        st.success("Chunking settings applied. Index rebuilt and chat reset.")
        st.rerun()

    # --- Retrieval (query) settings ---
    st.subheader("Retrieval Settings")
    st.slider(
        "Top-k results",
        min_value=1,
        max_value=20,
        value=st.session_state["_active_settings"]["k"],
        step=1,
        key="ui_k",
    )
    # Update k immediately; no index rebuild required
    st.session_state["_active_settings"]["k"] = st.session_state["ui_k"]

    # --- Model (generation) settings ---
    st.subheader("Model Settings")
    st.slider(
        "Model temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["_active_settings"]["temperature"],
        step=0.05,
        key="ui_temperature",
    )
    # Update temperature immediately; no index rebuild required
    st.session_state["_active_settings"]["temperature"] = st.session_state[
        "ui_temperature"
    ]


# ------------------------
# BUILD RETRIEVER + CHAIN
# ------------------------
active = st.session_state["_active_settings"]
vs = _build_vectorstore_cached(
    _settings_key(active["chunk_size"], active["chunk_overlap"])
)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": active["k"]},
)


def _format_docs_for_prompt(docs):
    out = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "source.md")
        out.append(f"[{i}] Source: {src}\n{d.page_content}")
    return "\n\n".join(out)


def _make_chain():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("human", "Question: {question}\n\nUse the context below.\n\n{context}"),
        ]
    )
    # TODO 3: build LLM with ChatGoogleGenerativeAI, using MODEL_NAME, API_KEY, and temperature from session_state
    # reference: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        api_key=API_KEY,
        temperature=st.session_state["_active_settings"]["temperature"],
    )

    def retrieve_context(x):
        # TODO 4: use retriever to get docs for x["question"]
        docs = 
        return _format_docs_for_prompt(docs)

    chain = (
        {
            "question": lambda x: x["question"],
            "context": retrieve_context,
            "history": lambda x: x["history"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


base_chain = _make_chain()

# ------------------------
# HISTORY + CHAT LOOP
# ------------------------
history = StreamlitChatMessageHistory(key="chat_history")
rag = RunnableWithMessageHistory(
    base_chain,
    lambda sid: history,
    input_messages_key="question",
    history_messages_key="history",
)

if len(history.messages) == 0:
    history.add_ai_message(
        "Hi! I’m your Yelp Bot. Ask me about a business based on the uploaded docs."
    )

# render past
for m in history.messages:
    _render_bubble("assistant" if isinstance(m, AIMessage) else "user", m.content)


# chat input
user_q = st.chat_input("Type your message…")
if user_q:
    _render_bubble("user", user_q)

    answer = rag.invoke(
        {"question": user_q},
        config={"configurable": {"session_id": "yelp-minimal"}},
    )
    _render_bubble("assistant", answer)
