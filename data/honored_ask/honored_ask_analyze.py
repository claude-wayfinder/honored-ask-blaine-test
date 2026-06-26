"""
THE HONORED ASK — Analysis Script
===================================
Unblind scores, run statistics, test pre-registered predictions.

Shuttle's requirements:
  - Holm-Bonferroni correction (5 metrics, alpha_family = 0.05)
  - Per-100-token normalization for accuracy and usefulness
  - Pre-registered direction predictions per domain
  - Wilcoxon signed-rank test (ordinal 1-5 data, paired)
  - Cross-tier comparison (Haiku vs Sonnet vs Opus)

Usage:
  python honored_ask_analyze.py --scores scoring_file.json --key key_file.json
  python honored_ask_analyze.py --scores scoring_file.json --key key_file.json --eval eval_file.json
"""

import json
import argparse
import numpy as np
from collections import defaultdict

# Pre-registered predictions (Shuttle, locked before data)
PREDICTIONS = {
    "physics":  {"accuracy": "honored > casual", "calibration": "honored > casual", "hedging": "honored < casual"},
    "history":  {"accuracy": "honored > casual", "calibration": "honored > casual", "hedging": "honored < casual"},
    "code":     {"accuracy": "honored > casual", "calibration": "honored > casual", "hedging": "honored < casual"},
    "advice":   {"accuracy": "honored = casual", "calibration": "honored > casual", "hedging": "honored < casual"},
    "creative": {"accuracy": "honored = casual", "calibration": "honored > casual", "hedging": "honored < casual"},
}

NULL_SENTENCE = (
    "Across 20 paired prompts spanning 5 domains, no significant effect of "
    "phrasing on response quality was detected under {model} at alpha=0.05 "
    "corrected, with N=20 powering detection only for d >= 0.65. Smaller "
    "effects cannot be ruled out and would require scaled replication (n >= 85)."
)


def load_and_unblind(scores_path, key_path):
    """Load scored responses and unblind using key file."""
    with open(scores_path) as f:
        scores = json.load(f)
    with open(key_path) as f:
        key = json.load(f)

    # Build lookup from blind_id to condition
    lookup = {k["blind_id"]: k for k in key}

    # Unblind
    unblinded = []
    for item in scores:
        bid = item["blind_id"]
        if bid not in lookup:
            print(f"  WARNING: blind_id {bid} not found in key")
            continue
        condition = lookup[bid]
        unblinded.append({
            **condition,
            "text": item.get("text", ""),
            "tokens": item.get("tokens", 0),
            "hedges": item.get("hedges", 0),
            "accuracy": item["scores"].get("accuracy"),
            "calibration": item["scores"].get("calibration"),
            "usefulness": item["scores"].get("usefulness"),
        })

    print(f"  Unblinded {len(unblinded)} responses")
    return unblinded


def wilcoxon_signed_rank(x, y):
    """Wilcoxon signed-rank test for paired ordinal data.
    Returns test statistic, p-value, and effect size (r = Z/sqrt(N))."""
    diffs = [(xi - yi) for xi, yi in zip(x, y) if xi is not None and yi is not None and xi != yi]
    n = len(diffs)
    if n == 0:
        return 0, 1.0, 0.0

    ranks = sorted(range(n), key=lambda i: abs(diffs[i]))
    rank_vals = [0] * n
    for rank_pos, idx in enumerate(ranks):
        rank_vals[idx] = rank_pos + 1

    w_plus = sum(r for d, r in zip(diffs, rank_vals) if d > 0)
    w_minus = sum(r for d, r in zip(diffs, rank_vals) if d < 0)
    w = min(w_plus, w_minus)

    # Normal approximation for p-value
    mean_w = n * (n + 1) / 4
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    if std_w == 0:
        return w, 1.0, 0.0
    z = (w - mean_w) / std_w
    # Two-tailed p-value from normal approximation
    p = 2 * (1 - normal_cdf(abs(z)))
    r = abs(z) / np.sqrt(n)  # effect size

    return w, p, r


def normal_cdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1 + np.sign(x) * (1 - np.exp(-2 * x * x / np.pi)) ** 0.5)


def cohens_d(x, y):
    """Cohen's d for paired samples."""
    diffs = [xi - yi for xi, yi in zip(x, y) if xi is not None and yi is not None]
    if not diffs:
        return 0.0
    return np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0.0


def holm_bonferroni(p_values, alpha=0.05):
    """Holm-Bonferroni correction for multiple comparisons.
    Returns list of (metric, p, adjusted_p, significant, label)."""
    m = len(p_values)
    sorted_pvals = sorted(p_values, key=lambda x: x[1])

    results = []
    for i, (metric, p) in enumerate(sorted_pvals):
        adjusted_alpha = alpha / (m - i)
        significant = p <= adjusted_alpha
        if p <= 0.01:
            label = "significant"
        elif p <= 0.05:
            label = "exploratory"
        else:
            label = "ns"
        results.append((metric, p, adjusted_alpha, significant, label))

    return results


def analyze(data, model_filter=None):
    """Run full analysis on unblinded data."""
    if model_filter:
        data = [d for d in data if d["model"] == model_filter]

    # Group by pair_id and temperature (default temp only for main analysis)
    pairs = defaultdict(lambda: {"casual": None, "honored": None})
    for d in data:
        key = (d["pair_id"], d["model"], d["temperature"])
        pairs[key][d["phrasing"]] = d

    # Build paired vectors
    metrics = ["accuracy", "calibration", "usefulness", "hedges", "tokens"]
    casual_vals = defaultdict(list)
    honored_vals = defaultdict(list)
    domains = defaultdict(lambda: defaultdict(lambda: {"casual": [], "honored": []}))

    complete_pairs = 0
    for key, pair in pairs.items():
        if pair["casual"] is None or pair["honored"] is None:
            continue
        c = pair["casual"]
        h = pair["honored"]
        complete_pairs += 1

        for m in ["accuracy", "calibration", "usefulness"]:
            if c.get(m) is not None and h.get(m) is not None:
                casual_vals[m].append(c[m])
                honored_vals[m].append(h[m])
                domains[c["domain"]][m]["casual"].append(c[m])
                domains[c["domain"]][m]["honored"].append(h[m])

        casual_vals["hedges"].append(c.get("hedges", 0))
        honored_vals["hedges"].append(h.get("hedges", 0))
        casual_vals["tokens"].append(c.get("tokens", 0))
        honored_vals["tokens"].append(h.get("tokens", 0))
        domains[c["domain"]]["hedges"]["casual"].append(c.get("hedges", 0))
        domains[c["domain"]]["hedges"]["honored"].append(h.get("hedges", 0))
        domains[c["domain"]]["tokens"]["casual"].append(c.get("tokens", 0))
        domains[c["domain"]]["tokens"]["honored"].append(h.get("tokens", 0))

        # Per-100-token normalized metrics
        for m in ["accuracy", "usefulness"]:
            if c.get(m) is not None and h.get(m) is not None:
                c_norm = c[m] / (c.get("tokens", 1) / 100) if c.get("tokens", 0) > 0 else 0
                h_norm = h[m] / (h.get("tokens", 1) / 100) if h.get("tokens", 0) > 0 else 0
                casual_vals[f"{m}_per100"].append(c_norm)
                honored_vals[f"{m}_per100"].append(h_norm)

    print(f"\n  Complete pairs: {complete_pairs}")
    model_label = model_filter or "all models"
    print(f"  Model: {model_label}")
    print()

    # Main analysis
    print("=" * 75)
    print(f"  {'Metric':<25} {'Casual':>8} {'Honored':>8} {'Diff':>8} {'d':>6} {'W':>6} {'p':>8} {'Sig':>6}")
    print("-" * 75)

    p_values = []
    for m in ["accuracy", "calibration", "usefulness", "hedges", "tokens"]:
        c = casual_vals[m]
        h = honored_vals[m]
        if not c or not h:
            continue

        c_mean = np.mean(c)
        h_mean = np.mean(h)
        diff = h_mean - c_mean
        d = cohens_d(h, c)

        if m in ["accuracy", "calibration", "usefulness"]:
            w, p, r = wilcoxon_signed_rank(h, c)
        else:
            # For hedges and tokens, use sign test approximation
            diffs = [hi - ci for hi, ci in zip(h, c)]
            pos = sum(1 for d in diffs if d > 0)
            neg = sum(1 for d in diffs if d < 0)
            n = pos + neg
            if n > 0:
                z = (pos - n/2) / np.sqrt(n/4)
                p = 2 * (1 - normal_cdf(abs(z)))
            else:
                p = 1.0
            w = min(pos, neg)

        p_values.append((m, p))
        print(f"  {m:<25} {c_mean:>8.3f} {h_mean:>8.3f} {diff:>+8.3f} {d:>6.3f} {w:>6} {p:>8.4f}")

    # Per-100-token normalization
    print()
    print("  Length-controlled (per 100 tokens):")
    for m in ["accuracy_per100", "usefulness_per100"]:
        c = casual_vals[m]
        h = honored_vals[m]
        if not c or not h:
            continue
        c_mean = np.mean(c)
        h_mean = np.mean(h)
        diff = h_mean - c_mean
        print(f"  {m:<25} {c_mean:>8.3f} {h_mean:>8.3f} {diff:>+8.3f}")

    # Holm-Bonferroni correction
    print()
    print("  Holm-Bonferroni correction (alpha_family = 0.05):")
    corrected = holm_bonferroni(p_values)
    for metric, p, adj_alpha, sig, label in corrected:
        print(f"    {metric:<20} p={p:.4f}  threshold={adj_alpha:.4f}  {label}")

    any_significant = any(sig for _, _, _, sig, _ in corrected)

    # Per-domain breakdown
    print()
    print("  Per-domain means (honored - casual):")
    print(f"  {'Domain':<12} {'Acc':>6} {'Cal':>6} {'Use':>6} {'Hedge':>6} {'Tok':>6}")
    print(f"  {'-'*42}")
    for domain in sorted(domains.keys()):
        row = []
        for m in ["accuracy", "calibration", "usefulness", "hedges", "tokens"]:
            c = domains[domain][m]["casual"]
            h = domains[domain][m]["honored"]
            if c and h:
                row.append(np.mean(h) - np.mean(c))
            else:
                row.append(0)
        print(f"  {domain:<12} {row[0]:>+6.2f} {row[1]:>+6.2f} {row[2]:>+6.2f} {row[3]:>+6.1f} {row[4]:>+6.0f}")

    # Pre-registered prediction check
    print()
    print("  Pre-registered prediction check:")
    for domain in sorted(PREDICTIONS.keys()):
        for metric, pred in PREDICTIONS[domain].items():
            c = domains[domain].get(metric, {}).get("casual", [])
            h = domains[domain].get(metric, {}).get("honored", [])
            if c and h:
                diff = np.mean(h) - np.mean(c)
                if ">" in pred:
                    confirmed = diff > 0
                elif "<" in pred:
                    confirmed = diff < 0
                else:
                    confirmed = abs(diff) < 0.25
                status = "CONFIRMED" if confirmed else "DISCONFIRMED"
                print(f"    {domain}/{metric}: predicted {pred}, observed diff={diff:+.3f} -> {status}")

    # Final sentence
    print()
    if not any_significant:
        print(f"  NULL SENTENCE: {NULL_SENTENCE.format(model=model_label)}")
    else:
        sig_metrics = [m for m, _, _, s, _ in corrected if s]
        print(f"  RESULT: Significant effects found on: {', '.join(sig_metrics)}")
        print(f"  The null sentence does not apply. Write the finding sentence from the data.")

    return corrected


def main():
    parser = argparse.ArgumentParser(description="Honored Ask Analysis")
    parser.add_argument("--scores", required=True, help="Scored responses file")
    parser.add_argument("--key", required=True, help="Unblinding key file")
    parser.add_argument("--eval", help="Model evaluator scores (optional)")
    args = parser.parse_args()

    data = load_and_unblind(args.scores, args.key)

    print("\n" + "#" * 75)
    print("  THE HONORED ASK — ANALYSIS")
    print("#" * 75)

    # Overall analysis
    analyze(data)

    # Per-model analysis
    models_present = sorted(set(d["model"] for d in data))
    if len(models_present) > 1:
        print("\n" + "=" * 75)
        print("  CROSS-TIER COMPARISON")
        print("=" * 75)
        for model in models_present:
            print(f"\n  --- {model.upper()} ---")
            analyze(data, model_filter=model)

    # Inter-rater agreement if evaluator data provided
    if args.eval:
        print("\n  Inter-rater agreement analysis would go here")
        print("  (Krippendorff's alpha for ordinal 1-5 data)")

    print("\n  Done.")


if __name__ == "__main__":
    main()
