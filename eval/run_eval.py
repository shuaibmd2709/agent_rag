import argparse
import importlib
import json
import re
import unicodedata
from pathlib import Path

SYSTEMS = {
    "baseline": "baseline.baseline_rag",
    "chunked": "src.pipeline_v1",
    "reranked": "src.pipeline_v2",
    "expanded": "src.pipeline_v3",
    "cited": "src.pipeline_v4",
}

REFUSALS = [
    "i don't know", "i don’t know", "i do not know", "does not mention",
    "do not contain", "does not contain", "not mentioned",
    "ich weiß nicht", "nicht enthalten", "nicht erwähnt",
]


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(?<=\d)[,\s](?=\d)", "", text)
    text = re.sub(r"\s*%", "%", text)
    return re.sub(r"\s+", " ", text).strip()


def facts_found(facts, text):
    t = norm(text)
    return [f for f in facts if norm(f) in t]

def cited_ok(q, answer):
    arts = q.get("source_article")
    if arts is None:
        return None
    arts = arts if isinstance(arts, list) else [arts]
    return any(
        re.search(rf"(article|artikel)\s*{a}\b", answer, re.I) for a in arts
    )

def score_question(q, result):
    answer = result["answer"]
    contexts = " ".join(c["text"] for c in result["contexts"])

    if not q["answerable"]:
        refused = any(norm(r) in norm(answer) for r in REFUSALS)
        return {"id": q["id"], "type": q["type"], "passed": refused,
                "context_hit": None, "answer_hit": None,"cited":None,}

    facts = q["key_facts"]
    in_answer = facts_found(facts, answer)
    in_context = facts_found(facts, contexts)
    required_ok = all(
        norm(r) in norm(answer) for r in q.get("required_key_facts", [])
    )
    needed = q.get("min_key_facts", len(facts))
    passed = len(in_answer) >= needed and required_ok
    return {
        "id": q["id"], "type": q["type"], "passed": passed,
        "context_hit": f"{len(in_context)}/{len(facts)}",
        "answer_hit": f"{len(in_answer)}/{len(facts)}",
        "cited":cited_ok(q,answer),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="dev1", choices=["dev1", "dev2", "all"])
    parser.add_argument("--system", default="baseline", choices=SYSTEMS)
    args = parser.parse_args()

    ask = importlib.import_module(SYSTEMS[args.system]).ask
    golden = json.loads(Path("eval/golden_set.json").read_text(encoding="utf-8"))
    if args.set != "all":
        golden = [q for q in golden if q.get("set", "dev1") == args.set]

    rows = []
    for q in golden:
        print(f"Running {q['id']}...")
        rows.append(score_question(q, ask(q["question"])))

    print(f"\n{'id':6}{'type':16}{'context_hit':14}{'answer_hit':12}{'cited':8}result")
    for r in rows:
        print(f"{r['id']:6}{r['type']:16}{str(r['context_hit']):14}"
              f"{str(r['answer_hit']):12}{str(r['cited']):8}"
              f"{'PASS' if r['passed'] else 'FAIL'}")
    passed = sum(r["passed"] for r in rows)
    print(f"\nScore: {passed}/{len(rows)}")

    out = Path(f"eval/results_{args.system}_{args.set}.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()