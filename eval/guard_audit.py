import json
from pathlib import Path

from src.guard import check
from src.llm import get_llm
from src.pipeline_v6 import retrieve
from src.pipeline_v7 import PROMPT_V5

golden = json.loads(Path("eval/golden_set.json").read_text(encoding="utf-8"))
hard_total = soft_total = num_total = answers = 0

for q in golden:
    if not q["answerable"]:
        continue
    chunks = retrieve(q["question"])
    context = "\n\n".join(c["text"] for c in chunks)
    answer = get_llm().invoke(
        PROMPT_V5.format(context=context, question=q["question"])
    ).content
    g = check(answer, context)

    answers += 1
    num_total += bool(g["unsupported_numbers"])
    hard_total += len(g["weak_lines"])
    soft_total += len(g["soft_lines"])

    print(f"\n{q['id']}  numbers={g['unsupported_numbers']}")
    for share, line in g["weak_lines"]:
        print(f"   HARD {share:.2f}  {line}")
    for share, line in g["soft_lines"]:
        print(f"   soft {share:.2f}  {line}")

print(f"\n{answers} answers: {num_total} with invented numbers, "
      f"{hard_total} hard lines, {soft_total} soft lines")