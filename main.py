"""Command-line entry point for the Russian minimal-pair corpus generator."""

from __future__ import annotations

import argparse
from pathlib import Path

from api_runner import ApiRunOptions, ProviderSpec, default_run_id, load_matrix_config, run_api_experiment
from exporter import export_pairs
from generator import generate_pairs
from metrics import compute_and_export_metrics
from review import finalize_review, prepare_review, review_report
from rules import RULES, normalize_rule_ids
from validator import validate_pairs


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description="Generate deterministic Russian minimal pairs for LLM grammar experiments."
    )
    parser.add_argument(
        "--mode",
        choices=("preview", "full"),
        default="preview",
        help="preview generates 3 pairs per rule by default; full generates 20.",
    )
    parser.add_argument(
        "--review-mode",
        choices=("prepare-review", "finalize-review", "review-report"),
        default=None,
        help="Run a manual-review workflow instead of generating a new corpus.",
    )
    parser.add_argument(
        "--api-mode",
        choices=("run", "run-matrix", "dry-run", "metrics"),
        default=None,
        help="Run the LLM API experiment workflow instead of generation/review.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Input CSV/JSON for review-mode or api-mode commands.",
    )
    parser.add_argument(
        "--num-per-rule",
        type=int,
        default=None,
        help="Override the number of pairs generated for each selected rule.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated CSV/JSON files.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="File name without extension. Defaults to demo_pairs in preview and full_pairs in full mode.",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=("csv", "json"),
        default=("csv", "json"),
        help="One or more export formats.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic candidate shuffling or API item sampling.",
    )
    parser.add_argument(
        "--rules",
        nargs="*",
        default=None,
        help="Optional rule ids, separated by spaces or commas.",
    )
    parser.add_argument(
        "--list-rules",
        action="store_true",
        help="Print available rules and exit.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip automatic validation before export.",
    )
    parser.add_argument(
        "--no-items",
        action="store_true",
        help="Do not export separate correct/incorrect API item files.",
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic", "deepseek"),
        default=None,
        help="Provider for --api-mode run/dry-run.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Single model name for --api-mode run/dry-run.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="One or more model names for --api-mode run/dry-run.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Prompt ids for --api-mode run/dry-run.",
    )
    parser.add_argument(
        "--matrix-config",
        default=None,
        help="JSON config for --api-mode run-matrix.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable run id for API outputs; defaults to a timestamped id.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an API run by skipping provider/model/prompt/item combinations already normalized.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit API items after deterministic sampling.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Low fixed decoding temperature for API experiments.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=200,
        help="Maximum output tokens for model answers.",
    )
    parser.add_argument(
        "--reasoning",
        default="",
        help="Optional reasoning effort passed to providers that support it.",
    )
    parser.add_argument(
        "--thinking",
        default="",
        help="Optional thinking mode for compatible providers such as DeepSeek.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry count for retryable API failures.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout in seconds for API calls.",
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=1.0,
        help="Base seconds for exponential retry backoff.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Reserved concurrency limit; current runner executes sequentially.",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.0,
        help="Optional pause in seconds between API requests.",
    )
    return parser


def main() -> int:
    """Run generation, validation, and export from the command line."""

    parser = build_parser()
    args = parser.parse_args()

    if args.list_rules:
        for rule in RULES:
            print(f"{rule.rule_id}: {rule.rule_name}")
        return 0

    if args.api_mode:
        return run_api_mode(args, parser)

    if args.review_mode:
        if not args.input:
            parser.error("--input is required when --review-mode is used")
        return run_review_mode(args)

    selected_rules = normalize_rule_ids(args.rules)
    num_per_rule = args.num_per_rule if args.num_per_rule is not None else (3 if args.mode == "preview" else 20)
    output_name = args.output_name or ("demo_pairs" if args.mode == "preview" else "full_pairs")
    output_dir = Path(args.output_dir or "./output")

    pairs = generate_pairs(num_per_rule=num_per_rule, rule_ids=selected_rules, seed=args.seed)

    if not args.no_validate:
        report = validate_pairs(pairs)
        if report.warnings:
            print("Validation warnings:")
            for warning in report.warnings:
                print(f"  - {warning}")
        if not report.ok:
            print("Validation errors:")
            for error in report.errors:
                print(f"  - {error}")
            return 1

    written_paths = export_pairs(
        pairs,
        output_dir=output_dir,
        formats=args.format,
        output_name=output_name,
        include_items=not args.no_items,
    )

    print(f"Generated {len(pairs)} pairs for {len(selected_rules)} rule(s).")
    for path in written_paths:
        print(f"Wrote {path}")
    print("Preview:")
    for pair in pairs[:5]:
        print(f"- [{pair.rule_id}] {pair.correct_sentence} -> {pair.incorrect_sentence}")
    return 0


def run_api_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Dispatch API experiment commands."""

    if args.api_mode == "metrics":
        if not args.input:
            parser.error("--input is required when --api-mode metrics is used")
        output_dir = Path(args.output_dir) if args.output_dir else Path(args.input).parent / "metrics"
        paths = compute_and_export_metrics(args.input, output_dir)
        print(f"Computed metrics from {args.input}.")
        for path in paths:
            print(f"Wrote {path}")
        return 0

    if args.api_mode == "run-matrix":
        if not args.matrix_config:
            parser.error("--matrix-config is required when --api-mode run-matrix is used")
        options = load_matrix_config(args.matrix_config, output_dir_override=args.output_dir, resume=args.resume)
        result = run_api_experiment(options)
        print_api_result(result)
        return 0

    if not args.input:
        parser.error("--input is required when --api-mode run/dry-run is used")
    if not args.provider:
        parser.error("--provider is required when --api-mode run/dry-run is used")
    models = selected_models(args)
    if not models:
        parser.error("--model or --models is required when --api-mode run/dry-run is used")

    run_id = args.run_id or default_run_id("dry_run" if args.api_mode == "dry-run" else "run")
    output_dir = Path(args.output_dir or Path("./runs") / run_id)
    provider_spec = ProviderSpec(
        name=args.provider,
        models=models,
        prompt_ids=args.prompts or ["direct_zero_shot"],
    )
    options = ApiRunOptions(
        input_path=Path(args.input),
        output_dir=output_dir,
        provider_specs=[provider_spec],
        run_id=run_id,
        limit=args.limit,
        seed=args.seed,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        reasoning=args.reasoning,
        thinking=args.thinking,
        dry_run=args.api_mode == "dry-run",
        resume=args.resume,
        max_retries=args.max_retries,
        timeout=args.timeout,
        backoff_base=args.backoff_base,
        max_concurrency=args.max_concurrency,
        sleep_between_requests=args.sleep_between_requests,
    )
    result = run_api_experiment(options)
    print_api_result(result)
    return 0


def selected_models(args: argparse.Namespace) -> list[str]:
    """Collect model names from --model and --models without duplicates."""

    models: list[str] = []
    if args.model:
        models.append(args.model)
    if args.models:
        models.extend(args.models)
    unique: list[str] = []
    for model in models:
        if model not in unique:
            unique.append(model)
    return unique


def print_api_result(result) -> None:
    """Print compact API run stats and output paths."""

    print(f"API run {result.run_id} finished.")
    print("API stats:")
    for key, value in result.stats.items():
        print(f"  {key}: {value}")
    for path in result.paths:
        print(f"Wrote {path}")


def run_review_mode(args: argparse.Namespace) -> int:
    """Dispatch review workflow commands and print compact stats."""

    input_path = Path(args.input)
    if args.review_mode == "prepare-review":
        output_dir = Path(args.output_dir or "./review")
        result = prepare_review(input_path, output_dir)
        print(f"Prepared review table from {input_path}.")
    elif args.review_mode == "finalize-review":
        output_dir = Path(args.output_dir or "./final")
        result = finalize_review(input_path, output_dir)
        print(f"Finalized approved clean corpus from {input_path}.")
    elif args.review_mode == "review-report":
        output_dir = Path(args.output_dir) if args.output_dir else input_path.parent
        result = review_report(input_path, output_dir)
        print(f"Built review report for {input_path}.")
    else:
        raise ValueError(f"Unsupported review mode: {args.review_mode}")

    for path in result.paths:
        print(f"Wrote {path}")
    print_review_stats(result.stats)
    return 0


def print_review_stats(stats: dict[str, object]) -> None:
    """Print the review summary requested by the CLI workflow."""

    print("Review stats:")
    print(f"  total_pairs: {stats.get('total_pairs', 0)}")
    print(f"  approved_pairs: {stats.get('approved_pairs', 0)}")
    print(f"  rejected_pairs: {stats.get('rejected_pairs', 0)}")
    print(f"  pending_pairs: {stats.get('pending_pairs', 0)}")
    print(f"  duplicate_text_pairs: {stats.get('duplicate_text_pairs', 0)}")
    print(f"  auto_flagged_pairs: {stats.get('auto_flagged_pairs', 0)}")
    approved_by_rule = stats.get("approved_by_rule", {})
    if approved_by_rule:
        print("  approved_by_rule:")
        for rule_id, count in dict(approved_by_rule).items():
            print(f"    {rule_id}: {count}")


if __name__ == "__main__":
    raise SystemExit(main())
