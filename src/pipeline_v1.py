import sys
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from baseline.baseline_rag import EMBED_MODEL, PROMPT, TOP_K
from src.chunk import build_chunks
from src.llm import get_llm

DB_DIR = "src/chroma_db"
COLLECTION = "chunked_v1"


@lru_cache(maxsize=1)
def _vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=DB_DIR,
    )


def build_index():
    if Path(DB_DIR).exists():
        print(f"{DB_DIR} already exists. Delete it first to rebuild.")
        return

    chunks = build_chunks()
    docs = [
        Document(
            page_content=c["text"],
            metadata={"doc": c["doc"], "article": c["article"], "page": c["page"]},
        )
        for c in chunks
    ]
    print(f"Chunks to index: {len(docs)}")

    store = _vectorstore()
    for i in range(0, len(docs), 500):
        store.add_documents(docs[i : i + 500])
        print(f"Indexed {min(i + 500, len(docs))}/{len(docs)}")


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
                "article": h.metadata.get("article"),
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
            print(f"  {c['doc']}, article {c['article']}, page index {c['page']}")