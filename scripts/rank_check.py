import sys

from src.pipeline_v1 import _vectorstore

question = sys.argv[1]
wanted_doc, wanted_article = sys.argv[2].split(":")

hits = _vectorstore().similarity_search_with_score(question, k=50)
found = 0
for rank, (doc, dist) in enumerate(hits, start=1):
    m = doc.metadata
    mark = ""
    if m["doc"] == wanted_doc and m["article"] == wanted_article:
        mark = "  <-- wanted"
        found += 1
    if rank <= 10 or mark:
        print(f"{rank:2}  dist={dist:.3f}  {m['doc']:10} article {m['article']:>4}{mark}")
print(f"\n{found} chunk(s) of {wanted_doc} article {wanted_article} in top 50")