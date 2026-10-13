import sys
import re

from src.guard import check, is_refusal, used_chunks
from src.llm import get_llm
from src.pipeline_v3 import _index
from src.pipeline_v6 import retrieve

PROMPT_V5 = """Answer the question using only the context below.
Copy the wording of the context as closely as possible.
Do not add numbers, dates or conditions that are not in the context.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question: {question}
Answer:"""

RETRY_NOTE = """

A previous draft of this answer contained content that is not in the context:
{problems}
Write the answer again. Use only statements and numbers that appear in the context."""


def sources_line(chunks, answer):
    if is_refusal(answer):
        return "Sources: none (the documents do not contain an answer)"
    seen = []
    for c in used_chunks(answer, chunks):
        s = f"{c['doc']} article {c['article']}"
        if s not in seen:
            seen.append(s)
    return "Sources: " + "; ".join(seen) if seen else "Sources: none matched"


def describe(guard):
    parts = []
    if guard["unsupported_numbers"]:
        parts.append("numbers not in the context: " + ", ".join(guard["unsupported_numbers"]))
    for _, line in guard["weak_lines"]:
        parts.append("unsupported line: " + line[:90])
    return "\n".join(parts)


def complete_passages(passages):
    by_article, position = _index()
    groups = {}
    for c in passages:
        loc = position.get(c["text"])
        if loc:
            groups.setdefault(loc[0], set()).add(loc[1])
    out = []
    for key, idxs in groups.items():
        last = min(max(idxs) + 1, len(by_article[key]) - 1)
        out.extend(by_article[key][j] for j in range(min(idxs), last + 1))
    return out

def ask(question: str) -> dict:
    chunks = retrieve(question)
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = PROMPT_V5.format(context=context, question=question)
    llm = get_llm()

    def verify(text):
        if is_refusal(text) and len(text.split()) <= 80 and not re.search(r"\d", text):
            return {"unsupported_numbers": [], "weak_lines": [],
                    "soft_lines": [], "flagged": False}
        return check(text, context)

    answer = llm.invoke(prompt).content
    guard = verify(answer)
    regenerated = False

    if guard["flagged"]:
        retry = llm.invoke(prompt + RETRY_NOTE.format(problems=describe(guard))).content
        retry_guard = verify(retry)
        regenerated = True
        if not retry_guard["flagged"]:
            answer, guard = retry, retry_guard

    warning = None
    if guard["flagged"]:
        warning = "Unverified content in the generated answer: " + describe(guard)
        if guard["unsupported_numbers"]:
            used = used_chunks(answer, chunks)
            if used:
                passages = complete_passages(used)
                answer = (
                    "I could not verify my own answer against the sources, "
                    "so here are the relevant passages instead:\n\n"
                    + "\n\n".join(c["text"] for c in passages)
                )
            else:
                answer = ("I don't know. The documents do not contain "
                          "a verifiable answer to this question.")
        else:
            drop = {line for _, line in guard["weak_lines"]}
            kept = [ln for ln in answer.splitlines() if ln.strip() not in drop]
            text = "\n".join(kept).strip()
            if not text:
                answer = ("I don't know. The documents do not contain "
                          "a verifiable answer to this question.")
            else:
                answer = (
                    text
                    + f"\n\n(Note: {len(drop)} statement(s) were removed because "
                    "they could not be verified against the sources.)"
                )

    return {
        "answer": answer,
        "sources": sources_line(chunks, answer),
        "guard": guard,
        "regenerated": regenerated,
        "warning": warning,
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
    if r["regenerated"]:
        print("(regenerated once after the guard flagged the first draft)")
    if r["warning"]:
        print("WARNING:", r["warning"])