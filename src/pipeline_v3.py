import sys
from functools import lru_cache

from baseline.baseline_rag import PROMPT, TOP_K
from src.chunk import build_chunks
from src.llm import get_llm
from src.pipeline_v1 import _vectorstore
from src.pipeline_v2 import CANDIDATES, _reranker

NEIGHBOURS = 1


@lru_cache(maxsize=1)
def _index():
    by_article = {}
    for c in build_chunks():
        by_article.setdefault((c["doc"], c["article"]), []).append(c)
    position = {}
    for key, items in by_article.items():
        for i, c in enumerate(items):
            position[c["text"]] = (key, i)
    return by_article, position


def retrieve(question: str):
    candidates = _vectorstore().similarity_search(question, k=CANDIDATES)
    scores = _reranker().predict([(question, c.page_content) for c in candidates])
    ranked = [
        c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    ][:TOP_K]

    by_article, position = _index()
    groups = {}
    for c in ranked:
        loc = position.get(c.page_content)
        if loc is None:
            continue
        key, i = loc
        lo = max(0, i - NEIGHBOURS)
        hi = min(len(by_article[key]), i + NEIGHBOURS + 1)
        groups.setdefault(key, set()).update(range(lo, hi))

    ordered = []
    for key, idxs in groups.items():
        ordered.extend(by_article[key][j] for j in sorted(idxs))
    return ordered


def ask(question: str) -> dict:
    chunks = retrieve(question)
    context = "\n\n".join(c["text"] for c in chunks)
    answer = get_llm().invoke(
        PROMPT.format(context=context, question=question)
    ).content
    return {
        "answer": answer,
        "contexts": [
            {"doc": c["doc"], "article": c["article"],
             "page": c["page"], "text": c["text"]}
            for c in chunks
        ],
    }


if __name__ == "__main__":
    result = ask(" ".join(sys.argv[1:]))
    print("\nANSWER:\n", result["answer"])
    print("\nRETRIEVED FROM:")
    for c in result["contexts"]:
        print(f"  {c['doc']}, article {c['article']}, page index {c['page']}")