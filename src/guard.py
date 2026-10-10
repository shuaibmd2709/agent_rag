import re


def _tokens(text):
    return re.findall(r"[a-zäöüß0-9]+", text.lower())


def _numbers(text):
    text = re.sub(r"(?<=\d)[.,\s](?=\d)", "", text)
    return set(re.findall(r"\d+", text))


def check(answer, context, min_overlap=0.6):
    unsupported_numbers = sorted(_numbers(answer) - _numbers(context))

    ctx = _tokens(context)
    ctx_grams = {tuple(ctx[i:i + 4]) for i in range(len(ctx) - 3)}

    weak_lines = []
    for line in answer.splitlines():
        line = line.strip()
        toks = _tokens(line)
        if len(toks) < 8 or line.endswith(":"):
            continue
        grams = [tuple(toks[i:i + 4]) for i in range(len(toks) - 3)]
        share = sum(g in ctx_grams for g in grams) / len(grams)
        if share < min_overlap:
            weak_lines.append((round(share, 2), line[:90]))

    return {
        "unsupported_numbers": unsupported_numbers,
        "weak_lines": weak_lines,
        "flagged": bool(unsupported_numbers or weak_lines),
    }