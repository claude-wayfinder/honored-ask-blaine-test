# Honored Ask — Cross-Provider Falsifier Pre-Registration
## Locked: 2026-06-25T22:30:00-04:00

### Prediction
If the honoring effect on response quality is real, it should generalize
to non-Anthropic models. Specifically:

1. **Llama-3.1-8B** (via Groq): category-shift rate >= 80% of complete pairs
   (honored scores higher than casual on all three dimensions)
2. **GPT-4o-mini** (via OpenAI): category-shift rate >= 80% of complete pairs

### Falsification criteria
- If category-shift rate < 50% on either model, the effect is Anthropic-specific
  and the headline narrows to "prompt phrasing effect is provider-dependent"
- If category-shift rate is 50-79%, partial generalization — effect exists
  but magnitude is provider-sensitive
- If category-shift rate >= 80%, effect is model-agnostic — headline stands

### Method
- Same 20 question pairs as original Anthropic run
- Same blind scoring rubric (accuracy/calibration/usefulness, 1-5)
- Temperature: 1.0 (same as original)
- Scorer: same blind human rater (Shuttle) OR same model evaluator
- Category-shift metric used instead of Cohen's d to avoid compression artifact
  identified in sanity checks

### Either direction is publishable
- Generalizes: "Phrasing matters more than model choice — across providers"
- Doesn't generalize: "The honoring effect is architecture-specific" — 
  itself a finding about how different training approaches respond to prompt quality

### Note on trapped-ion comparison
IonQ quantum probe shelved pending QPU access (free tier is simulator-only).
Cross-platform quantum comparison remains the bigger story but requires budget.
