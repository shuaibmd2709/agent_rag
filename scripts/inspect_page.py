import sys

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

pdf = sys.argv[1] if len(sys.argv) > 1 else "data/ai_act_en.pdf"
index = int(sys.argv[2]) if len(sys.argv) > 2 else 50

doc = pymupdf.open(pdf)
print(f"{pdf}: {len(doc)} pages, showing page index {index}\n")

for n, line in enumerate(doc[index].get_text().splitlines(), start=1):
    print(f"{n:3} | {line}")