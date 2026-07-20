"""Spectro CLI — multi-model cognitive spectrograph from the command line.

Usage:
    spectro "your question here"
    spectro "question" --models model1,model2,model3
    spectro "question" --format json
    spectro "question" --verbose
"""

from __future__ import annotations

import argparse
import json
import sys

from spectro.core import Spectrograph, DEFAULT_MODELS


def format_report(result, verbose: bool = False) -> str:
    """Format a SpectrumResult as a readable report."""
    lines = []
    lines.append("=" * 72)
    lines.append("SPECTRO — Cognitive Spectrum Report")
    lines.append("=" * 72)
    lines.append(f"\nPROMPT: {result.prompt}")
    lines.append(f"MODELS: {result.n_ok}/{result.n_models} responded OK")
    lines.append(f"TIME: {result.elapsed_ms:.0f}ms")
    lines.append(f"CONFIDENCE: {result.confidence:.0%}")
    lines.append("")

    # Spectrum bar
    bar_width = 40
    filled = int(result.confidence * bar_width)
    bar = "█" * filled + "░" * (bar_width - filled)
    lines.append(f"  [{bar}] {result.confidence:.0%} convergence")
    lines.append("")

    # Convergences (the coastline)
    lines.append("─" * 72)
    lines.append(f"CONVERGENCES ({len(result.convergences)}) — High-confidence territory")
    lines.append("─" * 72)
    for c in result.convergences[:15]:
        bar_filled = int(c["strength"] * 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        lines.append(f"  {bar} {c['concept']} ({c['agreement']}/{result.n_models})")
    if len(result.convergences) > 15:
        lines.append(f"  ... and {len(result.convergences) - 15} more")
    lines.append("")

    # Divergences (the interesting water)
    lines.append("─" * 72)
    lines.append(f"DIVERGENCES ({len(result.divergences)}) — Where models disagree")
    lines.append("─" * 72)
    for d in result.divergences[:10]:
        if "concept" in d:
            lines.append(f"  ◆ {d['concept']} — {d['note']}")
        elif "model_a" in d:
            short_a = d["model_a"].split("/")[-1][:20]
            short_b = d["model_b"].split("/")[-1][:20]
            lines.append(
                f"  ◆ {short_a} vs {short_b}: "
                f"{d.get('phrase_overlap', '?')} overlap — {d.get('note', '')}"
            )
    lines.append("")

    # Unique insights (what each model saw alone)
    lines.append("─" * 72)
    lines.append(f"UNIQUE INSIGHTS ({len(result.unique_insights)}) — Single-model perspectives")
    lines.append("─" * 72)
    # Group by model
    by_model = {}
    for u in result.unique_insights[:20]:
        model = u["model"].split("/")[-1]
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(u)

    for model, insights in by_model.items():
        lines.append(f"\n  ▸ {model}:")
        for u in insights[:5]:
            ctx = u.get("context", "")[:80]
            if ctx:
                lines.append(f"    · {u['concept']}: {ctx}...")
            else:
                lines.append(f"    · {u['concept']}")

    lines.append("")
    lines.append("=" * 72)

    # Verbose: show full responses
    if verbose:
        lines.append("\nFULL RESPONSES:")
        for resp in result.responses:
            lines.append(f"\n{'─' * 72}")
            short = resp.model.split("/")[-1]
            status = "OK" if resp.ok else f"ERROR: {resp.error}"
            lines.append(f"MODEL: {short} ({resp.elapsed_ms:.0f}ms, {resp.tokens} tok) [{status}]")
            lines.append("─" * 72)
            if resp.ok:
                lines.append(resp.content)
            lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="spectro",
        description="Multi-model cognitive spectrograph. Split the beam. Read the spectrum.",
    )
    parser.add_argument("prompt", help="The prompt/question to analyze")
    parser.add_argument(
        "--models", "-m",
        help="Comma-separated model list (default: 5-model ensemble)",
        default=None,
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full model responses",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens per response",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List default ensemble models and exit",
    )

    args = parser.parse_args()

    if args.list_models:
        print("Default ensemble (the repertory company):")
        for m in DEFAULT_MODELS:
            print(f"  {m}")
        return 0

    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]

    try:
        spec = Spectrograph(models=models)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result = spec.analyze(
        prompt=args.prompt,
        models=models,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(format_report(result, verbose=args.verbose))

    return 0


if __name__ == "__main__":
    sys.exit(main())
