import json
from pathlib import Path

from src.pipeline_v5 import ask

golden = json.loads(Path("eval/golden_set.json").read_text(encoding="utf-8"))

for q in golden:
    r = ask(q["question"])
    g = r["guard"]
    print(f"{q['id']}  flagged={g['flagged']}  "
          f"numbers={g['unsupported_numbers']}  weak_lines={len(g['weak_lines'])}")
    for share, line in g["weak_lines"]:
        print(f"      {share}  {line}")
    print("      ", r["sources"])