import sys

from src.guard import check
from src.llm import get_llm
from src.pipeline_v3 import retrieve

PROMPT_V5 = """Answer the question using only the context below.
Copy the wording of the context as closely as possible.
Do not add numbers, dates or conditions that are not in the context.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}
Answer:"""


def sources_line(chunks):
    seen = []
    for c in chunks:
        s = f"{c['doc']} article {c['article']}"
        if s not in seen:
            seen.append(s)
    return "Sources: " + "; ".join(seen)


def ask(question: str) -> dict:
    chunks = retrieve(question)
    context = "\n\n".join(c["text"] for c in chunks)
    answer = get_llm().invoke(
        PROMPT_V5.format(context=context, question=question)
    ).content
    return {
        "answer": answer,
        "sources": sources_line(chunks),
        "guard": check(answer, context),
        "contexts": [
            {"doc": c["doc"], "article": c["article"],
             "page": c["page"], "text": c["text"]}
            for c in chunks
        ],
    }


if __name__ == "__main__":
    r = ask(" ".join(sys.argv[1:]))
    print("\nANSWER:\n", r["answer"])
    print("\n" + r["sources"])
    print("GUARD:", r["guard"])