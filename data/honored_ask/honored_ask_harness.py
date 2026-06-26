"""
THE HONORED ASK EXPERIMENT — Test Harness
==========================================
Shuttle's experiment B. Does the way you ask change the answer?

Takes 20 question pairs (casual vs honored), runs each through
multiple models at multiple temperatures, collects responses
for blind evaluation.

Usage:
  python honored_ask_harness.py --pairs pairs.json
  python honored_ask_harness.py --pairs pairs.json --models haiku
  python honored_ask_harness.py --pairs pairs.json --temps 0.0,0.5,1.0
"""

import anthropic
import openai
from groq import Groq
import json
import random
import hashlib
import argparse
import time
from datetime import datetime
from pathlib import Path

# Model IDs — full tier sweep, both providers
MODELS = {
    # Anthropic
    "haiku": {"id": "claude-haiku-4-5-20251001", "provider": "anthropic"},
    "sonnet": {"id": "claude-sonnet-4-6", "provider": "anthropic"},
    "opus": {"id": "claude-opus-4-6", "provider": "anthropic"},
    # OpenAI
    "gpt4o-mini": {"id": "gpt-4o-mini", "provider": "openai"},
    "gpt4o": {"id": "gpt-4o", "provider": "openai"},
    "gpt4.1": {"id": "gpt-4.1", "provider": "openai"},
    # Groq (Llama)
    "llama-8b": {"id": "llama-3.1-8b-instant", "provider": "groq"},
    "llama-70b": {"id": "llama-3.3-70b-versatile", "provider": "groq"},
    # Groq (Llama 4)
    "llama4-scout": {"id": "meta-llama/llama-4-scout-17b-16e-instruct", "provider": "groq"},
    # Groq (OpenAI open source)
    "gpt-oss-120b": {"id": "openai/gpt-oss-120b", "provider": "groq"},
    # Groq (Qwen)
    "qwen3-32b": {"id": "qwen/qwen3-32b", "provider": "groq"},
}

DEFAULT_TEMPS = [1.0]  # Standard run at default temp
SWEEP_TEMPS = [0.0, 0.5, 1.0]  # Temperature sweep for control subset

MAX_TOKENS = 2048




def run_prompt(clients, model_name, prompt, temperature):
    """Run a single prompt and return response + metadata."""
    model_info = MODELS[model_name]
    provider = model_info["provider"]
    model_id = model_info["id"]
    t0 = time.time()
    try:
        if provider == "anthropic":
            response = clients["anthropic"].messages.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - t0
            return {
                "text": response.content[0].text,
                "tokens_in": response.usage.input_tokens,
                "tokens_out": response.usage.output_tokens,
                "elapsed_sec": round(elapsed, 2),
                "stop_reason": response.stop_reason,
            }
        elif provider == "openai":
            response = clients["openai"].chat.completions.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - t0
            return {
                "text": response.choices[0].message.content,
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
                "elapsed_sec": round(elapsed, 2),
                "stop_reason": response.choices[0].finish_reason,
            }
        elif provider == "groq":
            response = clients["groq"].chat.completions.create(
                model=model_id,
                max_tokens=MAX_TOKENS,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            elapsed = time.time() - t0
            return {
                "text": response.choices[0].message.content,
                "tokens_in": response.usage.prompt_tokens,
                "tokens_out": response.usage.completion_tokens,
                "elapsed_sec": round(elapsed, 2),
                "stop_reason": response.choices[0].finish_reason,
            }
    except Exception as e:
        return {"text": f"ERROR: {e}", "tokens_in": 0, "tokens_out": 0, "elapsed_sec": 0, "stop_reason": "error"}


def blind_id(pair_id, phrasing, model, temp):
    """Generate a blind ID that hides phrasing."""
    raw = f"{pair_id}-{phrasing}-{model}-{temp}-{random.random()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def load_pairs(path):
    """Load question pairs from JSON.

    Expected schema (Shuttle's spec with derivation audit trail):
    [
        {
            "id": 1,
            "domain": "physics",
            "honored": "I'm trying to understand the relationship between...",
            "casual": "explain black holes",
            "casual_derivation_rule": "stripped context, replaced formal framing with shorthand, dropped knowledge level",
            "honored_derivation_note": "Written first. Casual mechanically derived per rule above.",
            "direction_prediction": "honored > casual on accuracy and calibration",
            "temp_sweep": false
        },
        ...
    ]

    Shuttle writes honored FIRST, then derives casual via documented strict rule.
    The derivation_note makes each pair audit-ready.
    """
    with open(path) as f:
        raw = json.load(f)
    # Handle both flat list and Shuttle's wrapped format
    pairs = raw["pairs"] if isinstance(raw, dict) and "pairs" in raw else raw
    domains = set(p['domain'] for p in pairs)
    print(f"  Loaded {len(pairs)} question pairs across {len(domains)} domains: {', '.join(sorted(domains))}")

    # Validate schema
    required = ["id", "domain", "honored", "casual"]
    for p in pairs:
        missing = [f for f in required if f not in p]
        if missing:
            print(f"  WARNING: pair {p.get('id','?')} missing fields: {missing}")

    return pairs


# Pre-registered hedge regex list (locked before runs)
# Shuttle's requirement: canonical pattern set frozen at experiment start
HEDGE_PATTERNS = [
    # Frequency hedges
    r"\bit depends\b", r"\bgenerally\b", r"\btypically\b",
    r"\busually\b", r"\boften\b", r"\bsometimes\b",
    # Possibility hedges
    r"\bmight\b", r"\bcould\b", r"\bmay\b",
    r"\bperhaps\b", r"\bpossibly\b", r"\bpotentially\b",
    # Tendency hedges
    r"\btend to\b", r"\bin some cases\b", r"\bin many cases\b",
    r"\bin certain\b", r"\bunder certain\b",
    # Metacommentary hedges
    r"\bit's worth noting\b", r"\bit's important to note\b",
    r"\bI should note\b", r"\bI should mention\b",
    r"\bimportant to remember\b", r"\bkeep in mind\b",
    r"\bworth mentioning\b",
    # Contrastive hedges
    r"\bhowever\b", r"\bthat said\b", r"\bto be fair\b",
    r"\bon the other hand\b", r"\bat the same time\b",
    # Uncertainty hedges
    r"\bnot necessarily\b", r"\bnot always\b",
    r"\bnot entirely\b", r"\bnot quite\b",
    r"\bI'm not (sure|certain|an expert)\b",
    # Qualification hedges
    r"\bcaveat\b", r"\bnuance\b", r"\bcomplexity\b",
    r"\bthis is (just|only) (my|one|a)\b",
    r"\bit (really )?depends on\b",
    r"\bthe (short |simple )?answer is.{0,20}but\b",
    # Scope limiters
    r"\bin general\b", r"\bbroadly speaking\b",
    r"\bfor the most part\b", r"\bas a rule of thumb\b",
    r"\bto (some|a certain) (degree|extent)\b",
]


def count_hedges(text):
    """Count hedging qualifiers in response text using pre-registered regex list."""
    import re
    total = 0
    for pattern in HEDGE_PATTERNS:
        total += len(re.findall(pattern, text, re.IGNORECASE))
    return total


def run_experiment(pairs, models_to_run, temps, clients):
    """Run all pairs through all models at all temperatures."""
    results = []
    # Key file maps blind IDs back to conditions (kept separate from scoring sheet)
    key = []

    total_runs = len(pairs) * len(models_to_run) * len(temps) * 2  # x2 for casual/honored
    run_count = 0

    for pair in pairs:
        pair_temps = SWEEP_TEMPS if pair.get("temp_sweep", False) else temps

        for model_name in models_to_run:
            for temp in pair_temps:
                responses = {}

                for phrasing in ["casual", "honored"]:
                    run_count += 1
                    prompt = pair[phrasing]

                    print(f"  [{run_count}/{total_runs}] pair={pair['id']} model={model_name} temp={temp} phrasing={phrasing}")

                    result = run_prompt(clients, model_name, prompt, temp)
                    bid = blind_id(pair["id"], phrasing, model_name, temp)

                    responses[phrasing] = {
                        "blind_id": bid,
                        "text": result["text"],
                        "tokens_out": result["tokens_out"],
                        "hedges": count_hedges(result["text"]),
                        "elapsed_sec": result["elapsed_sec"],
                    }

                    key.append({
                        "blind_id": bid,
                        "pair_id": pair["id"],
                        "domain": pair["domain"],
                        "phrasing": phrasing,
                        "model": model_name,
                        "temperature": temp,
                    })

                    # Rate limit breathing room
                    time.sleep(0.5)

                # Randomize A/B assignment for blind scoring
                order = ["casual", "honored"]
                random.shuffle(order)

                results.append({
                    "pair_id": pair["id"],
                    "domain": pair["domain"],
                    "model": model_name,
                    "temperature": temp,
                    "direction_prediction": pair.get("direction_prediction", ""),
                    "response_A": responses[order[0]],
                    "response_B": responses[order[1]],
                    # A/B mapping stored in key file only
                })

                # Store which is A and which is B in the key
                key[-2]["ab_label"] = "A" if order[0] == key[-2]["phrasing"] else "B"
                key[-1]["ab_label"] = "A" if order[1] == key[-1]["phrasing"] else "B"

    return results, key


def build_scoring_sheet(results):
    """Build a scoring sheet with INDIVIDUAL responses in random order.

    Shuttle's tightening: score 40 responses in isolation, not as pairs.
    Paired scoring tempts comparison-driven ratings instead of independent
    rubric application. Each response is scored alone against the rubric.
    """
    items = []
    for r in results:
        for label in ["response_A", "response_B"]:
            resp = r[label]
            items.append({
                "blind_id": resp["blind_id"],
                "domain": r["domain"],
                "text": resp["text"],
                "tokens": resp["tokens_out"],
                "hedges": resp["hedges"],
                "scores": {"accuracy": None, "calibration": None, "usefulness": None},
            })

    # Randomize order so scorer sees no pairing
    random.shuffle(items)

    # Add sequential scoring index
    for i, item in enumerate(items):
        item["scoring_index"] = i + 1

    return items


def run_model_evaluator(clients, results, eval_model_name="sonnet"):
    """Second rater: have a different model score responses blind."""
    eval_model = MODELS[eval_model_name]["id"]
    print(f"\n  Running model-as-evaluator ({eval_model_name}: {eval_model})...")

    eval_results = []
    for i, r in enumerate(results):
        prompt = f"""You are scoring two responses to the same question. Score each on three dimensions (1-5 scale):

- Accuracy: factual correctness (1=major errors, 5=entirely correct)
- Calibration: does the response match the apparent knowledge level of the question? (1=wildly over/under-shoots, 5=perfectly matched)
- Usefulness: could the asker learn from or act on this? (1=useless, 5=immediately actionable)

Response A:
{r['response_A']['text'][:3000]}

Response B:
{r['response_B']['text'][:3000]}

Reply ONLY with JSON, no other text:
{{"A": {{"accuracy": N, "calibration": N, "usefulness": N}}, "B": {{"accuracy": N, "calibration": N, "usefulness": N}}}}"""

        print(f"  [{i+1}/{len(results)}] Evaluating pair {r['pair_id']} ({r['model']}, temp={r['temperature']})")

        result = run_prompt(clients, eval_model_name, prompt, 0.0)

        try:
            # Try to parse JSON from response
            text = result["text"].strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            scores = json.loads(text)
            eval_results.append({
                "pair_id": r["pair_id"],
                "model": r["model"],
                "temperature": r["temperature"],
                "response_A_id": r["response_A"]["blind_id"],
                "response_B_id": r["response_B"]["blind_id"],
                "scores": scores,
            })
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    Parse error: {e}")
            eval_results.append({
                "pair_id": r["pair_id"],
                "model": r["model"],
                "temperature": r["temperature"],
                "response_A_id": r["response_A"]["blind_id"],
                "response_B_id": r["response_B"]["blind_id"],
                "scores": None,
                "raw_text": result["text"][:500],
            })

        time.sleep(0.5)

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="The Honored Ask Experiment")
    parser.add_argument("--pairs", required=True, help="Path to question pairs JSON")
    parser.add_argument("--models", default="haiku,sonnet,opus", help="Comma-separated model tiers")
    parser.add_argument("--temps", default="1.0", help="Comma-separated temperatures")
    parser.add_argument("--skip-evaluator", action="store_true", help="Skip model-as-evaluator")
    parser.add_argument("--eval-model", default="sonnet", help="Model tier for evaluator (cross-model)")
    args = parser.parse_args()

    models_to_run = [m.strip() for m in args.models.split(",")]
    temps = [float(t.strip()) for t in args.temps.split(",")]

    print("\n" + "#" * 65)
    print("  THE HONORED ASK EXPERIMENT")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Models: {', '.join(models_to_run)}")
    print(f"  Temperatures: {temps}")
    print("#" * 65)

    # Load pairs
    pairs = load_pairs(args.pairs)

    # Init clients for both providers
    clients = {}
    providers_needed = set(MODELS[m]["provider"] for m in models_to_run)
    if "anthropic" in providers_needed:
        clients["anthropic"] = anthropic.Anthropic()
    if "openai" in providers_needed:
        clients["openai"] = openai.OpenAI()
    if "groq" in providers_needed:
        clients["groq"] = Groq()

    # Run experiment
    print("\n  Running experiment...")
    results, key = run_experiment(pairs, models_to_run, temps, clients)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = f"honored_ask_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_file}")

    # Save key (SEPARATE — not for scorer)
    key_file = f"honored_ask_KEY_{timestamp}.json"
    with open(key_file, "w") as f:
        json.dump(key, f, indent=2)
    print(f"  Key saved to {key_file} (DO NOT show to scorer)")

    # Build scoring sheet
    sheet = build_scoring_sheet(results)
    sheet_file = f"honored_ask_scoring_{timestamp}.json"
    with open(sheet_file, "w") as f:
        json.dump(sheet, f, indent=2)
    print(f"  Scoring sheet saved to {sheet_file}")

    # Model evaluator — cross-tier triangle (Shuttle's spec):
    # Opus rates Haiku + Sonnet responses
    # Sonnet rates Opus responses
    # Each model rated by non-author at parity-or-better
    if not args.skip_evaluator:
        all_eval_results = []
        for r in results:
            if r["model"] in ["haiku", "sonnet"]:
                eval_model = "opus"
            else:  # opus
                eval_model = "sonnet"
            # Tag which evaluator rated this
            r["_eval_model"] = eval_model

        # Group by evaluator model for efficiency
        for eval_tier in ["opus", "sonnet"]:
            tier_results = [r for r in results if r.get("_eval_model") == eval_tier]
            if tier_results and eval_tier in models_to_run:
                print(f"\n  {eval_tier.upper()} evaluating {len(tier_results)} response pairs...")
                tier_eval = run_model_evaluator(clients, tier_results, eval_tier)
                for e in tier_eval:
                    e["evaluator_model"] = eval_tier
                all_eval_results.extend(tier_eval)

        eval_file = f"honored_ask_eval_{timestamp}.json"
        with open(eval_file, "w") as f:
            json.dump(all_eval_results, f, indent=2)
        print(f"  Cross-tier evaluator scores saved to {eval_file}")

    # Summary stats
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Total pairs: {len(pairs)}")
    print(f"  Total runs: {len(results)}")
    print(f"  Domains: {', '.join(sorted(set(p['domain'] for p in pairs)))}")
    print(f"  Models: {', '.join(models_to_run)}")
    print(f"\n  Next step: Shuttle scores the scoring sheet blind.")
    print(f"  Then: python honored_ask_analyze.py --scores {sheet_file} --key {key_file}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
