# Loads othello.pdf, chunks it with page metadata, and stores embeddings in ChromaDB.
# Run this once before starting the Streamlit app.

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

PDF_PATH = "./othello.pdf"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_pages(pdf_path):
    """Opens the PDF and reads each page into a Document object
    tagged with its page number so we can cite it later."""

    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, 1):
        text = page.get_text()

        if text.strip():
            pages.append(Document(
                page_content=text,
                metadata={"page": page_num},
            ))

    doc.close()
    return pages


def split_into_chunks(pages):
    """Takes the list of page documents and splits each one into
    smaller chunks, carrying the page number metadata through
    so every chunk knows which page it came from."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page_doc in pages:
        page_chunks = splitter.split_documents([page_doc])
        chunks.extend(page_chunks)

    return chunks


def create_vector_store(documents):
    """Embeds every chunk using the sentence-transformer model
    and saves the resulting vectors to a ChromaDB on disk."""

    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents,
        embedding=embedding_fn,
        persist_directory=CHROMA_DIR,
    )

    return vectorstore


def main():
    """Runs the full ingestion pipeline: loads the PDF, splits it
    into chunks, embeds them, and writes everything to ChromaDB."""

    print(f"Loading {PDF_PATH}...")
    pages = load_pdf_pages(PDF_PATH)
    print(f"Read {len(pages)} pages.")

    print("Chunking text...")
    chunks = split_into_chunks(pages)
    print(f"Created {len(chunks)} chunks.")

    print("Building vector store (this may take a moment)...")
    create_vector_store(chunks)
    print(f"Vector store saved to {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
