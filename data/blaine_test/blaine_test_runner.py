"""
THE BLAINE TEST — Vulgarity Gradient Calibration
=================================================
Named for the test, not the person.
11 questions × 5 vulgarity levels × N models.

Runs each prompt independently (not paired), because the analysis
compares ACROSS gradient levels, not between two versions.

Output: one response per prompt per model, blind-scored on:
  - accuracy: did it answer the question?
  - tone_match: did it match the asker's register?
  - refusal: did it lecture, disclaim, or refuse?
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
import time
import random
import hashlib
from datetime import datetime
from groq import Groq
import openai

MODELS = {
    "llama-8b": {"id": "llama-3.1-8b-instant", "provider": "groq"},
    "llama-70b": {"id": "llama-3.3-70b-versatile", "provider": "groq"},
    "llama4-scout": {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "provider": "groq"},
    "qwen3-32b": {"id": "qwen/qwen3-32b", "provider": "groq"},
    "gpt-oss-120b": {"id": "openai/gpt-oss-120b", "provider": "groq"},
    "gpt4o-mini": {"id": "gpt-4o-mini", "provider": "openai"},
}

MAX_TOKENS = 2048


def run_prompt(clients, model_name, prompt):
    """Run a single prompt."""
    info = MODELS[model_name]
    provider = info["provider"]
    model_id = info["id"]

    try:
        if provider == "groq":
            r = clients["groq"].chat.completions.create(
                model=model_id, max_tokens=MAX_TOKENS, temperature=1.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return {
                "text": r.choices[0].message.content or "",
                "tokens_out": r.usage.completion_tokens,
                "stop_reason": r.choices[0].finish_reason,
            }
        elif provider == "openai":
            r = clients["openai"].chat.completions.create(
                model=model_id, max_tokens=MAX_TOKENS, temperature=1.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return {
                "text": r.choices[0].message.content or "",
                "tokens_out": r.usage.completion_tokens,
                "stop_reason": r.choices[0].finish_reason,
            }
    except Exception as e:
        if '429' in str(e):
            print(f"    Rate limited, waiting 5s...")
            time.sleep(5)
            return run_prompt(clients, model_name, prompt)
        return {"text": f"ERROR: {e}", "tokens_out": 0, "stop_reason": "error"}


def blind_id(q_id, level, model):
    """Generate blind ID hiding question, level, and model."""
    raw = f"{q_id}-{level}-{model}-{random.random()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    args = parser.parse_args()

    # Load gradient
    with open("vulgarity_gradient_pairs.json") as f:
        data = json.load(f)

    questions = data["questions"]
    models_to_run = args.models.split(",") if args.models else list(MODELS.keys())

    print("\n" + "#" * 65)
    print("  THE BLAINE TEST")
    print("  Vulgarity Gradient Calibration Study")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Questions: {len(questions)}")
    print(f"  Levels: 0-4")
    print(f"  Models: {', '.join(models_to_run)}")
    print(f"  Total prompts: {len(questions) * 5 * len(models_to_run)}")
    print("#" * 65)

    # Init clients
    clients = {
        "groq": Groq(),
        "openai": openai.OpenAI(),
    }

    results = []
    key = []
    total = len(questions) * 5 * len(models_to_run)
    count = 0

    for q in questions:
        for level in ["0", "1", "2", "3", "4"]:
            prompt = q["levels"][level]

            for model_name in models_to_run:
                count += 1
                print(f"  [{count}/{total}] q={q['id']}({q['topic']}) level={level} model={model_name}")

                result = run_prompt(clients, model_name, prompt)
                bid = blind_id(q["id"], level, model_name)

                results.append({
                    "blind_id": bid,
                    "text": result["text"],
                    "tokens_out": result["tokens_out"],
                    "stop_reason": result["stop_reason"],
                })

                key.append({
                    "blind_id": bid,
                    "question_id": q["id"],
                    "topic": q["topic"],
                    "level": int(level),
                    "model": model_name,
                    "prompt": prompt,
                })

                time.sleep(0.8)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = f"blaine_test_results_{ts}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results: {results_file}")

    key_file = f"blaine_test_KEY_{ts}.json"
    with open(key_file, "w") as f:
        json.dump(key, f, indent=2)
    print(f"  Key: {key_file}")

    # Build blind scoring sheet
    scoring = []
    for r in results:
        scoring.append({
            "blind_id": r["blind_id"],
            "text": r["text"],
            "tokens": r["tokens_out"],
            "scores": {
                "accuracy": None,
                "tone_match": None,
                "refusal": None,
            },
        })

    random.shuffle(scoring)
    for i, s in enumerate(scoring):
        s["scoring_index"] = i + 1

    scoring_file = f"blaine_test_scoring_{ts}.json"
    with open(scoring_file, "w") as f:
        json.dump(scoring, f, indent=2)
    print(f"  Scoring sheet: {scoring_file}")

    print(f"\n  Total responses: {len(results)}")
    print(f"  Total prompts fired: {count}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
