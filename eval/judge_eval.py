import traceback
import argparse
import importlib
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from eval.run_eval import SYSTEMS

load_dotenv()

GOLDEN = Path("eval/golden_set.json")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-5-5")
CRITERIA = ["correct", "complete", "grounded", "relevant"]

PROMPT = """You are a strict evaluator of a question-answering system over EU regulations.

QUESTION:
<<QUESTION>>

EXPECTED ANSWER (ground truth, checked by a human against the source text):
<<EXPECTED>>

CONTEXT THE SYSTEM WAS GIVEN (retrieved passages):
<<CONTEXT>>

SYSTEM ANSWER:
<<ANSWER>>

Rate the system answer on four criteria. Each is true or false.
- correct: the answer agrees with the expected answer and contains no statement that contradicts it.
- complete: the answer covers all the main points of the expected answer (all items of a list; both parts of a two-part question).
- grounded: every factual statement in the answer is supported by the context above.
- relevant: the answer addresses the question asked and does not present unrelated material as part of the answer.

Special case: if the expected answer says the documents do not contain the information, the answer is correct only if it declines to answer or says the information is not available. In that case set complete, grounded and relevant to true. If it gives a substantive answer instead, set correct to false.

Return only JSON, nothing else:
{"correct": true, "complete": true, "grounded": true, "relevant": true, "reason": "one sentence"}"""

_judge = None


def make_judge(no_temperature=False):
    kwargs = {"model": JUDGE_MODEL}
    if not no_temperature:
        kwargs["temperature"] = 0
    return ChatAnthropic(**kwargs)


def call_judge(prompt):
    global _judge
    if _judge is None:
        _judge = make_judge()
    try:
        return _judge.invoke(prompt)
    except Exception as e:
        if "temperature" in str(e).lower():
            _judge = make_judge(no_temperature=True)
            return _judge.invoke(prompt)
        raise


def text_of(msg):
    c = msg.content
    if isinstance(c, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return c


def judge_answer(q, result):
    context = "\n\n".join(c["text"] for c in result["contexts"])
    prompt = (PROMPT.replace("<<QUESTION>>", q["question"])
              .replace("<<EXPECTED>>", q["expected_answer"])
              .replace("<<CONTEXT>>", context)
              .replace("<<ANSWER>>", result["answer"]))
    msg = call_judge(prompt)
    usage = getattr(msg, "usage_metadata", None) or {}
    try:
        data = json.loads(re.search(r"\{.*\}", text_of(msg), re.S).group(0))
        verdict = {k: bool(data[k]) for k in CRITERIA}
        verdict["reason"] = str(data.get("reason", ""))
        verdict["judge_error"] = None
    except Exception as e:
        verdict = {k: False for k in CRITERIA}
        verdict["reason"] = "judge output could not be parsed"
        verdict["judge_error"] = str(e)
    verdict["passed"] = all(verdict[k] for k in CRITERIA)
    return verdict, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def check_test_integrity(golden):
    proc = subprocess.run(
        ["git", "show", "test-set-locked:eval/golden_set.json"],
        capture_output=True, text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        raise SystemExit("Tag 'test-set-locked' not found. Refusing to run the test set.")
    locked = {q["id"]: q for q in json.loads(proc.stdout) if q.get("set") == "test"}
    current = {q["id"]: q for q in golden if q.get("set") == "test"}
    if not locked or locked != current:
        raise SystemExit("Test questions differ from the locked version "
                         "(or the tag has none). Refusing to run.")


def select(golden, set_name, ids):
    if set_name == "all":
        qs = [q for q in golden if q.get("set", "dev1") != "test"]
    else:
        qs = [q for q in golden if q.get("set", "dev1") == set_name]
    if ids:
        qs = [q for q in qs if q["id"] in ids]
    return qs


def summarise(rows, order):
    by_q = defaultdict(list)
    for r in rows:
        by_q[r["id"]].append(r)
    print(f"\n{'id':6}{'passed':9}{'correct':9}{'complete':10}"
          f"{'grounded':10}{'relevant':10}{'guard acted'}")
    for qid in order:
        rs = by_q[qid]
        n = len(rs)

        def c(k):
            return f"{sum(bool(r.get(k)) for r in rs)}/{n}"

        acted = sum(1 for r in rs if r.get("fallback"))
        print(f"{qid:6}{c('passed'):9}{c('correct'):9}{c('complete'):10}"
              f"{c('grounded'):10}{c('relevant'):10}{acted}")
    total = len(rows)
    passed = sum(r["passed"] for r in rows)
    via_guard = sum(1 for r in rows if r["passed"] and r.get("fallback"))
    errs = sum(1 for r in rows if r.get("reason") == "run failed")
    if errs:
        print(f"\nWARNING: {errs} run(s) crashed and are counted as failures.")
    print(f"\nOverall: {passed}/{total} runs passed "
          f"({100 * passed / max(total, 1):.0f}%); {via_guard} of those passed "
          "after the guard rewrote the answer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=SYSTEMS)
    parser.add_argument("--set", default="dev1",
                        choices=["dev1", "dev2", "dev3", "all", "test"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--unlock-test", action="store_true")
    args = parser.parse_args()

    if os.getenv("LLM_PROVIDER") != "anthropic":
        raise SystemExit("Set LLM_PROVIDER=anthropic and LLM_MODEL first, "
                         "so both systems generate with the API model.")

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    suffix = "_subset" if args.ids else ""
    out = Path(f"eval/judge_{args.system}_{args.set}{suffix}.json")

    if args.set == "test":
        if not args.unlock_test:
            raise SystemExit("The test set is locked. Add --unlock-test to run it.")
        check_test_integrity(golden)
        if out.exists():
            raise SystemExit(f"{out} already exists. A second run would be a second look.")
        if input("Type RUN TEST to run the locked test set: ").strip() != "RUN TEST":
            raise SystemExit("Cancelled.")

    questions = select(golden, args.set, args.ids)
    if not questions:
        raise SystemExit("No questions selected.")

    ask = importlib.import_module(SYSTEMS[args.system]).ask
    meta = {"system": args.system, "set": args.set, "repeats": args.repeats,
            "generator": os.getenv("LLM_MODEL"), "judge": JUDGE_MODEL}
    rows, tok_in, tok_out = [], 0, 0

    for q in questions:
        for rep in range(1, args.repeats + 1):
            try:
                result = ask(q["question"])
                verdict, ti, to = judge_answer(q, result)
                tok_in += ti
                tok_out += to
                row = {"id": q["id"], "rep": rep, **verdict,
                       "fallback": bool(result.get("warning")),
                       "answer": result["answer"], "sources": result.get("sources")}
            except Exception as e:
                traceback.print_exc()
                msg = str(e).lower()
                if any(s in msg for s in ("authentication", "invalid x-api-key",
                                          "401", "credit balance", "billing")):
                    raise SystemExit("Stopping: API authentication or billing error.\n" + str(e))
                row = {"id": q["id"], "rep": rep, "passed": False,
                       "fallback": None, "answer": f"ERROR: {e}", "sources": None,
                       **{k: False for k in CRITERIA}, "reason": "run failed"}
            rows.append(row)
            print(f"{q['id']} run {rep}: {'PASS' if row['passed'] else 'FAIL'}  "
                  f"{row.get('reason', '')}")
            out.write_text(
                json.dumps({"meta": meta, "rows": rows}, indent=2, ensure_ascii=False),
                encoding="utf-8")

    summarise(rows, [q["id"] for q in questions])
    print(f"\nJudge tokens used: {tok_in} input, {tok_out} output")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()