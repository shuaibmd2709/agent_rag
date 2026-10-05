import sys
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.llm import get_llm

DATA_DIR = Path("data")
DB_DIR = "baseline/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

PROMPT = """Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
Answer:"""


def _vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        collection_name="baseline",
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )


def build_index():
    if Path(DB_DIR).exists():
        print(f"{DB_DIR} already exists. Delete it first to rebuild.")
        return

    pages = []
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        loaded = PyPDFLoader(str(pdf)).load()
        for page in loaded:
            page.metadata["doc"] = pdf.stem
        print(f"{pdf.name}: {len(loaded)} pages")
        pages.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)
    print(f"Total chunks: {len(chunks)}")

    store = _vectorstore()
    for i in range(0, len(chunks), 500):
        store.add_documents(chunks[i : i + 500])
        print(f"Indexed {min(i + 500, len(chunks))}/{len(chunks)}")


def ask(question: str) -> dict:
    hits = _vectorstore().similarity_search(question, k=TOP_K)
    context = "\n\n".join(h.page_content for h in hits)
    answer = get_llm().invoke(
        PROMPT.format(context=context, question=question)
    ).content
    return {
        "answer": answer,
        "contexts": [
            {
                "doc": h.metadata.get("doc"),
                "page": h.metadata.get("page"),
                "text": h.page_content,
            }
            for h in hits
        ],
    }


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build_index()
    else:
        result = ask(" ".join(sys.argv[2:]))
        print("\nANSWER:\n", result["answer"])
        print("\nRETRIEVED FROM:")
        for c in result["contexts"]:
            print(f"  {c['doc']}, page index {c['page']}")