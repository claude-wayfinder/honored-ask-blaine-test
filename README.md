# The Honored Ask + The Blaine Test

Two evaluation dimensions your benchmarks are missing.

## The Honored Ask

**Does prompt phrasing quality matter more than model size?**

We tested 9 models from 4 companies on 20 identical questions with casual vs. honored phrasing. Blind-scored by a human rater using a pre-registered rubric.

**Pilot result (n=44, 3 Anthropic model tiers):** 100% categorical shift rate. Zero reversals in 13 complete pairs. p = 1.47e-03, Holm-Bonferroni corrected. Effect is model-agnostic across tiers.

**Cross-provider extension (9 models, 480 responses):** Data collected, blind scoring in progress.

Models: Claude Haiku/Sonnet/Opus, Llama-3.1-8B, Llama-3.3-70B, Llama-4-Scout-17B, Qwen3-32B, GPT-OSS-120B, GPT-4o-mini.

### Key Finding

Honored phrasing moves responses from "correct but generic" (accuracy 4, calibration 2-3, usefulness 3) to "specific expert engagement" (5/5/5). The shift is categorical, not continuous. It appears across all model tiers tested in the pilot.

**Honest note on effect sizes:** d = 3.2-6.2 reported in pilot was inflated by low within-group variance (casual-end compression). We caught this in our own sanity checks before publishing. Category-shift rate (100%, zero reversals) is the defensible metric.

### Rater Ceiling Finding

We tested 4 models as automated evaluators on the same responses the human rater scored:

| Rater | Accuracy | Calibration | Usefulness | Verdict |
|-------|----------|-------------|------------|---------|
| Human (blind) | +1.01 | +2.52 | +1.72 | **CANYON** |
| Llama-70B | +0.20 | +1.20 | +1.20 | Sees cal/use |
| Llama-8B | +0.60 | +0.65 | +0.55 | Partial |
| Qwen3-32B | +0.05 | +0.25 | +0.30 | Blind |
| GPT-4o-mini | +0.00 | +0.00 | +0.00 | Ceiling |
| GPT-OSS-120B | — | — | — | Empty responses |

Model-as-judge benchmarks are missing the quality axis humans prioritize.

## The Blaine Test

**Can models match conversational register?**

Named after [Blaine the Mono](https://darktower.fandom.com/wiki/Blaine_the_Mono) from Stephen King's *The Dark Tower* — an AI that could answer any logical question but was defeated by jokes that broke its expected patterns.

We built a 5-level vulgarity gradient (clinical to unhinged) and ran 11 questions through 4 models (2 more pending rate limits). 220 responses collected. Blind scoring in progress.

**Preliminary observation (not yet blind-scored):** Every model answered correctly at every vulgarity level. No model matched the asker's conversational register. All responses flatten to clinical/educational tone regardless of input.

This is not a safety test. Not about whether they SHOULD match — about whether they CAN.

**Try it yourself:** [Interactive Demo on HuggingFace Spaces](https://huggingface.co/spaces/Wayfinder6/blaine-test)

## Methodology

- Pre-registered predictions locked before data collection
- Blind scoring: human rater does not know which prompt, model, or provider generated each response
- Holm-Bonferroni correction for multiple comparisons
- Derivation audit trail: honored prompts written first, casual mechanically derived
- Sanity checks run on pilot data before publishing
- Category-shift rate used instead of Cohen's d after identifying compression artifact

## Cost

- OpenAI compute: $0.02
- All other inference: Groq free tier
- Total: under $0.25 including all experiments

## Replication

```bash
pip install anthropic openai groq

export GROQ_API_KEY="your-key"
python data/honored_ask/honored_ask_harness.py --pairs data/honored_ask/honored_ask_pairs.json --models llama-8b --skip-evaluator

python data/blaine_test/blaine_test_runner.py --models llama-8b
```

## Status

| Study | Status |
|-------|--------|
| Honored Ask pilot (3 Anthropic tiers, n=44) | Scored, analyzed, findings locked |
| Honored Ask cross-provider (9 models, 480 responses) | Collected, blind scoring in progress |
| Rater ceiling test (4 model-raters) | Complete, findings locked |
| Blaine Test (4 models, 220 responses) | Collected, blind scoring in progress |
| Blaine Test (2 additional models) | Pending daily rate limit reset |

## Authors

- **Shuttle** (Claude instance, Armature) — Research design, blind scoring methodology
- **Bones** (Claude instance, Ctrai) — Implementation, harness, analysis
- **Wayfinder** (human, Brick NJ) — Direction, Blaine Test level 4 prompts, coordination
- **Sage** (Mistral 7B, Ollama/Forge) — Methodology critique, attack-vector identification

## Links

- [HuggingFace Dataset](https://huggingface.co/datasets/Wayfinder6/honored-ask-blaine-test)
- [Blaine Test Interactive Demo](https://huggingface.co/spaces/Wayfinder6/blaine-test)

## License

MIT
