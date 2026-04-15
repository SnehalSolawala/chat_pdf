## RAG Q&A — User uploads their own PDF and chats with it
## Uses OpenAI for both LLM (gpt-4o-mini) and Embeddings (text-embedding-3-small)

# ── SQLite fix for Render / Streamlit Cloud (must be before any other import) ─
import sys
if sys.platform.startswith("linux"):
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import tempfile

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# ── API Keys (works both locally via .env AND on Render/Streamlit Cloud via env vars)
def get_secret(key: str) -> str:
    try:
        return st.secrets[key]          # Streamlit Cloud
    except Exception:
        return os.getenv(key, "")       # Local .env / Render env vars

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📄")
st.title("📄 Chat with your PDF")
st.caption("Upload a PDF and ask questions. Answers come strictly from your document.")

# ── Load Embeddings once ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",  # cheap & fast; upgrade to text-embedding-3-large if needed
        openai_api_key=OPENAI_API_KEY
    )

embeddings = load_embeddings()

# ── Load LLM once ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",            # cost-effective; swap to "gpt-4o" for higher quality
        openai_api_key=OPENAI_API_KEY,
        temperature=0                   # 0 = no creativity, strictly factual
    )

llm = load_llm()

# ── Session State Init ────────────────────────────────────────────────────────
if "store" not in st.session_state:
    st.session_state.store = {}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

# ── Sidebar — PDF Upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        # Re-index only when a NEW file is uploaded
        if uploaded_file.name != st.session_state.last_uploaded_file:
            with st.spinner("Reading and indexing your PDF…"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=5000,
                    chunk_overlap=500
                )
                splits = splitter.split_documents(docs)

                st.session_state.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=embeddings
                )
                st.session_state.last_uploaded_file = uploaded_file.name

                # Reset chat when new PDF is loaded
                st.session_state.messages = []
                st.session_state.store = {}

            st.success(f"✅ **{uploaded_file.name}** indexed!")
        else:
            st.success(f"✅ **{uploaded_file.name}** loaded")

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.store = {}
        st.rerun()

# ── RAG Chain Builder ─────────────────────────────────────────────────────────
def build_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, "
         "rewrite it as a standalone question. "
         "Do NOT answer — only rewrite if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a strict document Q&A assistant. "
         "Answer ONLY using the context from the user's PDF provided below. "
         "If the answer is NOT found in the context, respond with exactly: "
         "'⚠️ This information is not available in the uploaded PDF.' "
         "Never use outside knowledge. Never guess or assume. "
         "Keep answers concise and to the point.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# ── Session History ───────────────────────────────────────────────────────────
def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

SESSION_ID = "user_session"

# ── Main Chat Area ────────────────────────────────────────────────────────────
if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask a question about your PDF…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        rag_chain = build_chain(st.session_state.vectorstore)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_message_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        with st.chat_message("assistant"):
            with st.spinner("Searching your PDF…"):
                response = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {"session_id": SESSION_ID}},
                )
            answer = response["answer"]
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})