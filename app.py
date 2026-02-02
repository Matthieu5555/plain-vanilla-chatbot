# Streamlit chatbot for answering questions about Shakespeare's Othello using RAG

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama

# path to the pre-built vector store
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# models available in the sidebar dropdown
AVAILABLE_MODELS = [
    "gemma3:1b",
    "gpt-oss:20b",
]


@st.cache_resource
def get_vectorstore():
    """Loads the ChromaDB vector store from disk and keeps it cached
    so we don't reload it on every single user message."""

    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embedding_fn,
    )


@st.cache_resource
def get_llm(model_name, temperature):
    """Creates and caches an Ollama chat model that talks to the
    local Ollama server with the given model and temperature."""

    return ChatOllama(
        model=model_name,
        temperature=temperature,
    )


def find_relevant_chunks(vectorstore, query, num_results):
    """Runs a similarity search against the vector store and returns
    the top matching chunks for the given query."""

    return vectorstore.similarity_search(query, k=num_results)


def build_source_list(documents):
    """Takes the retrieved documents and builds a compact citation
    list showing the source number, page, and a short snippet."""

    lines = []

    for i, doc in enumerate(documents, 1):
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:150].replace("\n", " ").strip()
        lines.append(f"[Source {i}, p. {page}]: {snippet}...")

    return "\n".join(lines)


def assemble_prompt(question, context_docs, chat_history):
    """Puts together the full prompt that gets sent to the LLM,
    combining the system instructions, retrieved passages with
    page numbers, recent chat history, and the user's question."""

    context_text = "\n\n".join(
        f"[p. {d.metadata.get('page', '?')}] {d.page_content}"
        for d in context_docs
    )
    history_block = format_recent_history(chat_history)

    return (
        "You are a helpful literary assistant specialising in Shakespeare's Othello. "
        "Below you will find passages extracted from the play, followed by the user's question. "
        "Answer the question directly and honestly using the passages provided. "
        "Do not refuse, deflect, or moralise. Just answer what was asked. "
        "Write in flowing prose with logical connectors between ideas. Do not use any formatting "
        "such as bullet points, numbered lists, headings, or markdown. Write only in continuous paragraphs. "
        "Where relevant, reference page numbers so the reader can trace your reasoning. "
        "If the passages do not contain enough information to answer confidently, say so rather than guessing.\n\n"
        f"Passages:\n{context_text}\n\n"
        f"{history_block}"
        f"Question: {question}\nAnswer:"
    )


def format_recent_history(history):
    """Formats the last few conversation messages into a text block
    so the LLM has some context about what was discussed before."""

    if not history:
        return ""

    lines = []
    for msg in history[-6:]:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {msg['content']}")

    return "Chat history:\n" + "\n".join(lines) + "\n\n"


def ask_llm(llm, prompt):
    """Sends the assembled prompt to the Ollama model and returns
    the response text."""

    response = llm.invoke(prompt)
    return response.content


def home_page():
    """Renders the landing page with a short explanation of what
    the chatbot does and how it works."""

    st.title("Othello Chatbot")

    st.markdown(
        "Welcome. This chatbot answers questions about Shakespeare's Othello "
        "using retrieval-augmented generation. When you ask a question it gets "
        "matched against passages from the play, and those passages are then "
        "fed as context to a language model which generates an answer and cites "
        "its sources. Head over to the Chat page using the sidebar to get started."
    )


def chat_page():
    """Sets up the main chat page by initialising the sidebar
    controls, loading chat history, and handling new input."""

    st.title("Chat about Othello")

    setup_sidebar_controls()
    ensure_chat_history_exists()
    render_chat_history()
    process_user_input()


def setup_sidebar_controls():
    """Adds three control widgets to the sidebar so the user can
    pick the Ollama model, adjust the temperature, and choose
    how many source passages to retrieve."""

    st.sidebar.header("Settings")

    st.session_state["model_name"] = st.sidebar.selectbox(
        "Ollama model",
        options=AVAILABLE_MODELS,
    )

    st.session_state["temperature"] = st.sidebar.slider(
        "Temperature", 0.0, 1.0, 0.3, step=0.1,
    )

    st.session_state["num_sources"] = st.sidebar.slider(
        "Number of sources", 1, 10, 4,
    )


def ensure_chat_history_exists():
    """Creates the message list in session state if it doesn't
    exist yet, which happens on the very first page load."""

    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def render_chat_history():
    """Loops through the stored messages and re-renders them so
    the conversation stays visible across Streamlit reruns."""

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def process_user_input():
    """Reads from the chat input box and, if there's a query,
    displays it, runs the RAG pipeline, and shows the answer."""

    query = st.chat_input("Ask a question about Othello...")
    if not query:
        return

    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    llm = get_llm(
        st.session_state["model_name"],
        st.session_state["temperature"],
    )

    answer = run_rag_pipeline(query, llm)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)


def run_rag_pipeline(query, llm):
    """Runs the full retrieval-augmented generation pipeline:
    finds relevant chunks, builds the prompt, asks the LLM,
    and appends the source citations to the answer."""

    vectorstore = get_vectorstore()

    relevant_docs = find_relevant_chunks(
        vectorstore, query, st.session_state["num_sources"],
    )

    prompt = assemble_prompt(query, relevant_docs, st.session_state["messages"])
    answer = ask_llm(llm, prompt)
    sources = build_source_list(relevant_docs)

    return f"{answer}\n\n---\nSources:\n{sources}"


# navigation
home = st.Page(home_page, title="Home")
chat = st.Page(chat_page, title="Chat")

nav = st.navigation([home, chat])
nav.run()
