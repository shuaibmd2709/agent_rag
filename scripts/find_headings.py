import re
import sys

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

pdf = sys.argv[1]
doc = pymupdf.open(pdf)
pattern = re.compile(r"^(Article|Artikel)\s*\d+\b")

shown = 0
for i, page in enumerate(doc):
    for line in page.get_text().splitlines():
        s = line.strip()
        if pattern.match(s) and len(s) <= 40:
            print(i, repr(s))
            shown += 1
    if shown >= 10:
        break
print(f"\n{shown} matching lines shown")