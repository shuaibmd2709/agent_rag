import re
import sys
from collections import Counter
from pathlib import Path

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

DATA_DIR = Path("data")

ARTICLE_RE = re.compile(r"^(?:Article|Artikel) (\d+)$")
ANNEX_RE = re.compile(r"^(?:ANNEX|ANHANG) ([IVXLC]+)$")
MARKER_RE = re.compile(r"^(\d{1,3}\.|\([0-9a-z]{1,4}\))$")


def read_pages(pdf_path):
    doc = pymupdf.open(pdf_path)
    return [
        [ln.strip() for ln in page.get_text().splitlines() if ln.strip()]
        for page in doc
    ]


def shape(line):
    return re.sub(r"\d+", "#", line)


def find_junk_shapes(pages, min_share=0.5):
    counts = Counter()
    for lines in pages:
        counts.update({
            shape(ln) for ln in lines
            if not MARKER_RE.match(ln)
            and not ARTICLE_RE.match(ln)
            and not ANNEX_RE.match(ln)
        })
    return {s for s, c in counts.items() if c >= min_share * len(pages)}


def split_sections(pages, doc_name):
    junk = find_junk_shapes(pages)
    sections = []
    current = {"doc": doc_name, "article": "preamble", "page": 0, "lines": []}
    expected = 1

    for page_no, lines in enumerate(pages):
        for line in lines:
            if not MARKER_RE.match(line) and shape(line) in junk:
                continue

            art = ARTICLE_RE.match(line)
            annex = ANNEX_RE.match(line)

            if art and int(art.group(1)) == expected:
                sections.append(current)
                current = {"doc": doc_name, "article": str(expected),
                           "page": page_no, "lines": [line]}
                expected += 1
            elif annex:
                sections.append(current)
                current = {"doc": doc_name, "article": f"annex-{annex.group(1)}",
                           "page": page_no, "lines": [line]}
            else:
                current["lines"].append(line)

    sections.append(current)
    for s in sections:
        s["text"] = "\n".join(s.pop("lines"))
    return sections


def load_all():
    return {
        pdf.stem: split_sections(read_pages(pdf), pdf.stem)
        for pdf in sorted(DATA_DIR.glob("*.pdf"))
    }


if __name__ == "__main__":
    data = load_all()
    for name, sections in data.items():
        arts = [s for s in sections if s["article"].isdigit()]
        annexes = [s for s in sections if s["article"].startswith("annex")]
        longest = sorted(arts, key=lambda s: len(s["text"]), reverse=True)[:3]
        print(f"{name}: {len(arts)} articles, {len(annexes)} annexes, "
              f"preamble {len(sections[0]['text'])} chars")
        print("   longest:", ", ".join(
            f"Art. {s['article']} ({len(s['text'])} chars)" for s in longest))

    if len(sys.argv) > 2:
        doc, art = sys.argv[1], sys.argv[2]
        for s in data[doc]:
            if s["article"] == art:
                print(f"\n--- {doc}, article {art}, starts on page index {s['page']} ---")
                print(s["text"][:1500])