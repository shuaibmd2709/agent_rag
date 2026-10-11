import re
import sys

from baseline.baseline_rag import PROMPT, TOP_K
from src.llm import get_llm
from src.pipeline_v1 import _vectorstore
from src.pipeline_v2 import CANDIDATES, _reranker
from src.pipeline_v3 import NEIGHBOURS, _index
from src.pipeline_v3 import retrieve as retrieve_all

REGULATIONS = {
    "ai_act": {
        "docs": ["ai_act_en", "ai_act_de"],
        "patterns": [r"\bai act\b", r"artificial intelligence act",
                     r"\bki-verordnung\b", r"2024/1689"],
    },
    "dora": {
        "docs": ["dora_en", "dora_de"],
        "patterns": [r"\bdora\b", r"digital operational resilience act",
                     r"2022/2554"],
    },
}
PER_REGULATION = 3


def detect(question):
    q = question.lower()
    return [
        name for name, r in REGULATIONS.items()
        if any(re.search(p, q) for p in r["patterns"])
    ]


def expand(selected):
    by_article, position = _index()
    groups = {}
    for c in selected:
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


def retrieve(question):
    regs = detect(question)
    if not regs:
        return retrieve_all(question)

    keep = TOP_K if len(regs) == 1 else PER_REGULATION
    selected = []
    for name in regs:
        cands = _vectorstore().similarity_search(
            question, k=CANDIDATES,
            filter={"doc": {"$in": REGULATIONS[name]["docs"]}},
        )
        if not cands:
            continue
        scores = _reranker().predict([(question, c.page_content) for c in cands])
        ranked = [
            c for _, c in sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)
        ]
        selected.extend(ranked[:keep])
    return expand(selected)


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