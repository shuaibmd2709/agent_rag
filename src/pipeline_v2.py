import sys
from functools import lru_cache

from sentence_transformers import CrossEncoder

from baseline.baseline_rag import PROMPT, TOP_K
from src.llm import get_llm
from src.pipeline_v1 import _vectorstore

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
CANDIDATES = 30


@lru_cache(maxsize=1)
def _reranker():
    return CrossEncoder(RERANK_MODEL, max_length=512)


def retrieve(question: str, k: int = TOP_K):
    candidates = _vectorstore().similarity_search(question, k=CANDIDATES)
    scores = _reranker().predict([(question, c.page_content) for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[:k]]


def ask(question: str) -> dict:
    hits = retrieve(question)
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
    result = ask(" ".join(sys.argv[1:]))
    print("\nANSWER:\n", result["answer"])
    print("\nRETRIEVED FROM:")
    for c in result["contexts"]:
        print(f"  {c['doc']}, article {c['article']}, page index {c['page']}")