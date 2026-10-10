import argparse
import importlib
import json
from pathlib import Path

from eval.run_eval import SYSTEMS

parser = argparse.ArgumentParser()
parser.add_argument("--system", default="cited")
parser.add_argument("ids", nargs="+")
args = parser.parse_args()

ask = importlib.import_module(SYSTEMS[args.system]).ask
golden = {
    q["id"]: q
    for q in json.loads(Path("eval/golden_set.json").read_text(encoding="utf-8"))
}

for qid in args.ids:
    q = golden[qid]
    result = ask(q["question"])
    print(f"\n===== {qid} (expected article: {q.get('source_article')}) =====")
    print(result["answer"])
    print("-- retrieved:",
          sorted({f"{c['doc']}:{c['article']}" for c in result["contexts"]}))