"""MiniMax-M3.1 accuracy evals on the Inferact/mqa protocol, standalone (needs only `datasets` and Pillow).

Reproduces what our numbers come from (mqa `--api-type chat_completions --think true` + the M3.1 template kwargs):
  aime26           mqa `--task aime26 --task-args '{"step_by_step": false}'`, 16 epochs, max_tokens 128000,
                   every request up front; score = exact match on the last \\boxed{} (avg@16), plus pass@16.
  mmmu_pro         mqa branch m31/provider-verification `--task mmmu_pro --task-args '{"subset": "all"}'`
                   (vision + standard (10 options), 3460 questions), 1 epoch, max_tokens 192000, concurrency 256;
                   scores strict-match (last line 'Answer: X') and flexible-match, overall and per subset.
All requests: temperature 1.0, top_p 0.95, chat_template_kwargs {thinking_mode: adaptive, thinking: true,
enable_thinking: true}, thinking {type: enabled}, timeout 60000 s. Prompts, image encoding (RGB -> JPEG, text first)
and scoring are mqa's, copied. thinking_mode adaptive: the 929ced46 chat template is adaptive only (it ignores
thinking_mode), and with adaptive the older template (a60dfa7f, AMD's Quark MXFP4 export) renders the same prompt.

    python3 eval.py --base-url http://localhost:28870 --model minimax-m31-flash --task aime26 --out ./out
    python3 eval.py --base-url http://localhost:28870 --model minimax-m31-flash --task mmmu_pro --out ./out

Results: <out>/<task>.jsonl (one record per request; a rerun skips the records already there) and
<out>/<task>_summary.json.
"""

import argparse
import ast
import base64
import concurrent.futures as cf
import io
import json
import os
import re
import sys
import threading
import urllib.request

# ---- AIME (mqa/tasks/aime.py) ----

MATH_ANSWER_FORMAT_PROMPT_NO_STEP_BY_STEP = """Please solve the math problem and provide your final answer in the following format:

1. Wrap your final answer with \\boxed{}
2. The content inside \\boxed{} should be concise - only the number or mathematical expression, no extra text

Examples of correct format:
- \\boxed{42}

Examples of incorrect format:
- The answer is 42. (missing \\boxed{})
- \\boxed{The answer is 42} (extra text inside)
- 42 (missing \\boxed{})\n\n"""

AIME_DATASETS = {"aime26": "math-ai/aime26"}


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def extract_answer_number(response):
    indices = [pos for pos, char in enumerate(response) if char == "$"]
    answer = response if len(indices) <= 1 else response[indices[0] + 1 : indices[-1]]
    boxed_answer = last_boxed_only_string(response)
    if boxed_answer is not None:
        try:
            answer = remove_boxed(boxed_answer)
        except (AssertionError, IndexError):
            pass
    return answer


def aime_exact_match(prediction, reference):
    return 1.0 if extract_answer_number(reference) == extract_answer_number(prediction) else 0.0


def aime_items(task):
    from datasets import load_dataset

    items = []
    for idx, row in enumerate(load_dataset(AIME_DATASETS[task], split="test")):
        prompt = MATH_ANSWER_FORMAT_PROMPT_NO_STEP_BY_STEP + "Problem: " + row["problem"] + "\nAnswer:"
        answer = str(row["answer"]) if row["answer"] is not None else ""
        items.append({"key": str(idx), "messages": [{"role": "user", "content": prompt}], "ref": answer})
    return items


# ---- MMMU-Pro (mqa/tasks/mmmu_pro.py, branch m31/provider-verification) ----

VISION_PROMPT = (
    "Answer with the option letter from the given choices directly. "
    "The last line of your response should be of the following format: "
    "'Answer: $LETTER' (without quotes) where LETTER is one of options."
)
_IMAGE_TAG_RE = re.compile(r"<image\s+(\d+)>")


def parse_options(options):
    return "\n".join(f"{chr(ord('A') + i)}. {option}" for i, option in enumerate(options))


def _option_list(options):
    return ast.literal_eval(options) if isinstance(options, str) else list(options)


def strict_match(pred, ref):
    pred = pred.replace("*", "").strip()
    if not pred:
        return 0.0
    pred = pred.splitlines()[-1]
    match = re.search(r"Answer:\s*([A-J])", pred, re.IGNORECASE)
    if match:
        return 1.0 if match.group(1).upper() == ref.upper() else 0.0
    return 0.0


def extract_answer_letter(response):
    if not response:
        return None
    text = response.replace("*", "")
    for pattern in (r"answer is \(?([A-J])\)?", r"answer:\s*\(?([A-J])\)?", r"\b([A-J])\b\s*$", r"\(([A-J])\)"):
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].upper()
    return None


def flexible_match(pred, ref):
    return 1.0 if extract_answer_letter(pred) == ref.strip().upper() else 0.0


def _messages_with_images(prompt, images):
    content = [{"type": "text", "text": prompt}]
    for image in images:
        with io.BytesIO() as buf:
            image.convert("RGB").save(buf, format="JPEG")
            data = base64.b64encode(buf.getvalue()).decode("utf-8")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}})
    return [{"role": "user", "content": content}]


def mmmu_items(subset):
    from datasets import load_dataset

    items = []
    if subset in ("vision", "all"):
        for row in load_dataset("MMMU/MMMU_Pro", name="vision", split="test"):
            prompt = (f"{row['question']}\n" if row.get("question") else "") + (
                f"{parse_options(ast.literal_eval(row['options']))}\n{VISION_PROMPT}"
            )
            items.append({"key": f"vision/{row['id']}", "subset": "vision", "ref": row["answer"],
                          "messages": _messages_with_images(prompt, [row["image"]])})
    if subset in ("standard", "all"):
        for row in load_dataset("MMMU/MMMU_Pro", name="standard (10 options)", split="test"):
            order = [int(n) for n in _IMAGE_TAG_RE.findall(row["question"])]
            question = _IMAGE_TAG_RE.sub("<image>", row["question"])
            prompt = f"{question}\n{parse_options(_option_list(row['options']))}\n{VISION_PROMPT}"
            images = [row[f"image_{n}"] for n in order]
            if any(image is None for image in images):
                raise ValueError(f"{row['id']}: question references an image the row lacks")
            items.append({"key": f"standard/{row['id']}", "subset": "standard", "ref": row["answer"],
                          "messages": _messages_with_images(prompt, images)})
    return items


def mmmu_answer_text(message):
    """mqa's parse_result: the visible content, or reasoning_content when the content has no parseable letter."""
    content = message.get("content") or ""
    if extract_answer_letter(content) is None and message.get("reasoning_content"):
        return message["reasoning_content"]
    return content


# ---- driver ----

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["aime26", "mmmu_pro"])
    ap.add_argument("--subset", default="all", choices=["all", "vision", "standard"], help="mmmu_pro only")
    ap.add_argument("--epochs", type=int, help="default 16 for AIME, 1 for MMMU-Pro")
    ap.add_argument("--max-tokens", type=int, help="default 128000 for AIME, 192000 for MMMU-Pro")
    ap.add_argument("--concurrency", type=int, help="default: every AIME request at once (480), 256 for MMMU-Pro")
    ap.add_argument("--out", default="./m31_eval_out")
    args = ap.parse_args()

    aime = args.task != "mmmu_pro"
    epochs = args.epochs or (16 if aime else 1)
    max_tokens = args.max_tokens or (128000 if aime else 192000)
    items = aime_items(args.task) if aime else mmmu_items(args.subset)
    jobs = [(e, it) for e in range(epochs) for it in items]
    concurrency = args.concurrency or (len(jobs) if aime else 256)

    os.makedirs(args.out, exist_ok=True)
    name = args.task if aime else f"mmmu_pro_{args.subset}"
    path = os.path.join(args.out, f"{name}.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path):
            rec = json.loads(line)
            done[(rec["epoch"], rec["key"])] = rec
    todo = [(e, it) for e, it in jobs if (e, it["key"]) not in done]
    print(f"{name}: {len(jobs)} requests, {len(done)} already in {path}, {len(todo)} to send, concurrency {concurrency}",
          file=sys.stderr)

    url = args.base_url.rstrip("/") + "/v1/chat/completions"
    lock = threading.Lock()

    def run(job):
        epoch, it = job
        body = {"model": args.model, "messages": it["messages"], "max_tokens": max_tokens, "temperature": 1.0,
                "top_p": 0.95,
                "chat_template_kwargs": {"thinking_mode": "adaptive", "enable_thinking": True, "thinking": True},
                "thinking": {"type": "enabled"}}
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=60000))
        choice = out["choices"][0]
        msg = choice["message"]
        if aime:
            answer = msg.get("content") or ""
            scores = {"exact_match": aime_exact_match(answer, it["ref"])}
        else:
            answer = mmmu_answer_text(msg)
            scores = {"strict-match": strict_match(answer, it["ref"]), "flexible-match": flexible_match(answer, it["ref"])}
        rec = {"epoch": epoch, "key": it["key"], "subset": it.get("subset"), "ref": it["ref"], "scores": scores,
               "finish_reason": choice.get("finish_reason"), "completion_tokens": out["usage"]["completion_tokens"],
               "content": msg.get("content"), "reasoning": msg.get("reasoning_content") or msg.get("reasoning")}
        with lock:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done[(epoch, it["key"])] = rec
            if len(done) % 50 == 0 or len(done) == len(jobs):
                print(f"{len(done)}/{len(jobs)}", file=sys.stderr)

    with cf.ThreadPoolExecutor(concurrency) as ex:
        for fut in cf.as_completed([ex.submit(run, j) for j in todo]):
            if fut.exception() is not None:
                print(f"request failed: {fut.exception()!r} (rerun to retry)", file=sys.stderr)

    recs = list(done.values())
    mean = lambda vals: round(sum(vals) / len(vals), 4) if vals else None  # noqa: E731
    summary = {"task": name, "model": args.model, "requests": len(recs), "expected": len(jobs), "epochs": epochs,
               "max_tokens": max_tokens,
               "truncated": sum(r["finish_reason"] == "length" for r in recs),
               "think_end_in_content": sum("</mm:think>" in (r["content"] or "") for r in recs)}
    if aime:
        summary["exact_match"] = mean([r["scores"]["exact_match"] for r in recs])
        solved = {}
        for r in recs:
            solved[r["key"]] = max(solved.get(r["key"], 0.0), r["scores"]["exact_match"])
        summary[f"pass@{epochs}"] = mean(list(solved.values()))
    else:
        for score in ("strict-match", "flexible-match"):
            summary[score] = mean([r["scores"][score] for r in recs])
            for sub in ("vision", "standard"):
                vals = [r["scores"][score] for r in recs if r["subset"] == sub]
                if vals:
                    summary[f"{score}/{sub}"] = mean(vals)
    with open(os.path.join(args.out, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

