# Othello Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Shakespeare's Othello. It downloads the full text of the play from Project Gutenberg, splits it into chunks, generates vector embeddings, and stores them in a ChromaDB database. When you ask a question, the app embeds your query, retrieves the most relevant passages, and feeds them as context to a local language model running through Ollama. The model then generates an answer and cites its sources.

## Prerequisites

You need three things installed before running this project: Python, Ollama, and pip.

**Python 3.10+** is required. You can check your version by running `python --version` in a terminal. If you don't have Python, download it from https://www.python.org/downloads/.

**Ollama** is the local LLM server that runs the language model on your machine. Install it from https://ollama.com/download. Once installed, open a terminal and pull at least one of the supported models:

```
ollama pull llama3.2
```

The app also supports `mistral` and `gemma3`, which you can pull the same way. Ollama must be running in the background when you use the chatbot (it starts automatically after install on most systems, or you can run `ollama serve`).

**pip** comes bundled with Python. You will use it to install the project's dependencies.

## Setup

First, install the Python dependencies:

```
pip install -r requirements.txt
```

Next, build the vector database. This downloads the play from Project Gutenberg, chunks the text, generates embeddings using the all-MiniLM-L6-v2 sentence-transformer model, and stores everything in a local ChromaDB folder:

```
python create_vector_db.py
```

Finally, start the Streamlit app:

```
streamlit run app.py
```

A browser tab will open with the chatbot. Use the sidebar to navigate between the Home page and the Chat page.

## How It Works

The `create_vector_db.py` script fetches the full text of Othello from Project Gutenberg, strips the Gutenberg header and footer, splits the play into chunks of roughly 300 to 500 words, and embeds each chunk using the all-MiniLM-L6-v2 model from HuggingFace. The resulting vectors are stored in a local ChromaDB database on disk.

The `app.py` script runs a Streamlit web application with two pages. The Home page explains what the chatbot does. The Chat page provides a conversational interface where you type questions about Othello. Your question is embedded into a vector, compared against the stored chunks using similarity search, and the top matching passages are assembled into a prompt along with recent chat history. That prompt is sent to a local Ollama model, which generates an answer and cites which passages it drew from.

## Configuration

The sidebar on the Chat page offers three widgets. The model selector lets you choose between llama3.2, mistral, and gemma3 (you must have pulled the model in Ollama first). The temperature slider controls how creative or deterministic the responses are. The source count slider sets how many passages are retrieved from the vector database for each query.
