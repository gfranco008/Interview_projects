#!/usr/bin/env python3
"""Validate the synthetic dataset generator configuration and outputs."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from generate_dataset import (
    build_output_columns,
    generate_rows,
    load_config,
    prepare_config,
    validate_config,
    validate_distributions,
    validate_rows,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate generate_dataset outputs")
    parser.add_argument("--rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--extra-columns", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--end-timestamp",
        type=str,
        default=None,
        help="ISO timestamp to anchor the 90-day window (default: now)",
    )
    args = parser.parse_args()

    raw_config = load_config(args.config)
    validate_config(raw_config)
    config = prepare_config(raw_config)

    end_ts = None
    if args.end_timestamp:
        end_ts = datetime.fromisoformat(args.end_timestamp)

    rows = generate_rows(args.rows, args.seed, end_ts, config, args.extra_columns)
    output_columns = build_output_columns(args.extra_columns)

    warnings = []
    warnings.extend(validate_rows(rows, output_columns))
    warnings.extend(validate_distributions(rows, config, args.rows))

    if warnings:
        print("Validation warnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("Validation passed with no warnings.")

    if warnings and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
