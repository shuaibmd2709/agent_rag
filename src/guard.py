import re


def _tokens(text):
    return re.findall(r"[a-zäöüß0-9]+", text.lower())


def _numbers(text):
    text = re.sub(r"(?<=\d)[.,\s](?=\d)", "", text)
    return set(re.findall(r"\d+", text))


def check(answer, context, hard=0.2, soft=0.6):
    unsupported_numbers = sorted(_numbers(answer) - _numbers(context))

    ctx = _tokens(context)
    ctx_grams = {tuple(ctx[i:i + 4]) for i in range(len(ctx) - 3)}

    weak_lines, soft_lines = [], []
    for line in answer.splitlines():
        line = line.strip()
        toks = _tokens(line)
        if len(toks) < 8 or line.endswith(":"):
            continue
        grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        share = sum(g in ctx_grams for g in grams) / len(grams)
        if share < hard:
            weak_lines.append((round(share, 2), line))
        elif share < soft:
            soft_lines.append((round(share, 2), line))

    return {
        "unsupported_numbers": unsupported_numbers,
        "weak_lines": weak_lines,
        "soft_lines": soft_lines,
        "flagged": bool(unsupported_numbers or weak_lines),
    }

REFUSAL_MARKERS = [
    "i don't know", "i do not know", "does not contain", "do not contain",
    "not mentioned", "ich weiß nicht", "nicht enthalten",
]


def is_refusal(answer):
    a = answer.lower().replace("’", "'")
    return any(m in a for m in REFUSAL_MARKERS)


def used_chunks(answer, chunks, min_shared=3, rel=0.3):
    toks = _tokens(answer)
    a_grams = {tuple(toks[i:i + 4]) for i in range(len(toks) - 3)}
    scored = []
    for c in chunks:
        ct = _tokens(c["text"])
        c_grams = {tuple(ct[i:i + 4]) for i in range(len(ct) - 3)}
        scored.append((len(a_grams & c_grams), c))
    best = max((s for s, _ in scored), default=0)
    return [c for s, c in scored if s >= min_shared and s >= rel * best]