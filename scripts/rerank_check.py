import sys

from src.pipeline_v1 import _vectorstore
from src.pipeline_v2 import CANDIDATES, _reranker

question = sys.argv[1]
wanted_doc, wanted_article = sys.argv[2].split(":")

cands = _vectorstore().similarity_search(question, k=CANDIDATES)
scores = _reranker().predict([(question, c.page_content) for c in cands])
ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)

for rank, (score, c) in enumerate(ranked, start=1):
    m = c.metadata
    wanted = m["doc"] == wanted_doc and m["article"] == wanted_article
    if rank <= 8 or wanted:
        lines = c.page_content.split("\n")
        first = lines[1][:70] if len(lines) > 1 else ""
        flag = "  <-- wanted" if wanted else ""
        print(f"{rank:2}  score={score:.3f}  {m['doc']:8} art {m['article']:>3}  {first}{flag}")

