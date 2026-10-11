import json
from pathlib import Path

from src.guard import check
from src.pipeline_v6 import ask

golden = json.loads(Path("eval/golden_set.json").read_text(encoding="utf-8"))

for q in golden:
    if not q["answerable"]:
        continue
    r = ask(q["question"])
    context = "\n\n".join(c["text"] for c in r["contexts"])
    g = check(r["answer"], context)
    print(f"\n{q['id']}  numbers={g['unsupported_numbers']}")
    for share, line in g["weak_lines"]:
        print(f"   HARD {share:.2f}  {line[:110]}")
    for share, line in g["soft_lines"]:
        print(f"   soft {share:.2f}  {line[:110]}")