import re

from src.ingest import load_all

HARD_MARKER = re.compile(r"^\(\d+\)(\s|$)")
SOFT_MARKER = re.compile(r"^(\d{1,3}\.$|\([a-z]{1,2}\)\s)")
SOFT_MIN = 500
HARD_MAX = 1100
HARD_MIN = 100

DOC_LABEL = {
    "ai_act_en": "EU AI Act (English)",
    "ai_act_de": "EU AI Act (German)",
    "dora_en": "DORA (English)",
    "dora_de": "DORA (German)",
}


def chunk_section(section):
    lines = section["text"].split("\n")
    doc = section["doc"]
    title = lines[1] if len(lines) > 1 else ""
    label = f"{DOC_LABEL.get(doc, doc)} | {lines[0]} | {title}"
    body = lines[2:]

    chunks, cur, cur_len = [], [], 0

    def flush():
        nonlocal cur, cur_len
        if cur:
            chunks.append("\n".join(cur))
        cur, cur_len = [], 0

    for line in body:
        n = len(line) + 1
        if cur_len > 0 and (
            (HARD_MARKER.match(line) and cur_len >= HARD_MIN)
            or (SOFT_MARKER.match(line) and cur_len >= SOFT_MIN)
            or cur_len + n > HARD_MAX
        ):
            flush()
        cur.append(line)
        cur_len += n
    flush()

    return [
        {
            "doc": doc,
            "article": section["article"],
            "title": title,
            "page": section["page"],
            "text": f"[{label}]\n{body_text}",
        }
        for body_text in chunks
    ]


def build_chunks(include_recitals=False):
    all_chunks = []
    for doc, sections in load_all().items():
        for section in sections:
            if section["article"] == "preamble" and not include_recitals:
                continue
            all_chunks.extend(chunk_section(section))
    return all_chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Total chunks: {len(chunks)}")
    for doc in DOC_LABEL:
        sizes = [len(c["text"]) for c in chunks if c["doc"] == doc]
        print(f"  {doc}: {len(sizes)} chunks, "
              f"avg {sum(sizes) // len(sizes)} chars, max {max(sizes)}")

    probes = [
        ("dora_en", r"major ICT-related incident.{1,3}means"),
        ("ai_act_en", r"serious incident.{1,3}means"),
        ("ai_act_en", r"EUR 35 000 000"),
    ]
    for doc, pattern in probes:
        hits = [c for c in chunks if c["doc"] == doc and re.search(pattern, c["text"])]
        print(f"\n=== {doc}, '{pattern}': {len(hits)} matching chunk(s) ===")
        if hits:
            print(hits[0]["text"])