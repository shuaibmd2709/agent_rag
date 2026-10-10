import sys

from src.llm import get_llm
from src.pipeline_v3 import retrieve

PROMPT_V4 = """Answer the question using only the context below.
Every chunk starts with a label like [Document | Article N | Title].
After each statement, cite its source in brackets, like [Article 16].
If the context does not contain the answer, say you don't know.
If the context contains only part of a list or definition, say that the list may be incomplete.

Context:
{context}

Question: {question}
Answer:"""


def ask(question: str) -> dict:
    chunks = retrieve(question)
    context = "\n\n".join(c["text"] for c in chunks)
    answer = get_llm().invoke(
        PROMPT_V4.format(context=context, question=question)
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
