import re
import sys

from src.ingest import load_all

phrase = sys.argv[1]
for doc, sections in load_all().items():
    for s in sections:
        for m in re.finditer(re.escape(phrase), s["text"], re.I):
            label = "RECITALS" if s["article"] == "preamble" else f"article {s['article']}"
            snippet = s["text"][max(0, m.start() - 250): m.end() + 350].replace("\n", " ")
            print(f"[{doc} | {label}]\n   ...{snippet}...\n")