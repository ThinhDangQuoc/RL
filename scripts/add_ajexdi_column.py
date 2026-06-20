#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add ajexdi=1.0 column to processed.csv, normalize datadate to YYYYMMDD, "
            "and save to processed2.csv."
        )
    )
    parser.add_argument(
        "--input",
        default="data/processed.csv",
        help="Input CSV path (default: data/processed.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed2.csv",
        help="Output CSV path (default: data/processed2.csv)",
    )
    parser.add_argument(
        "--index-col",
        default=0,
        type=int,
        help="Index column to use when reading CSV (default: 0)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, index_col=args.index_col)

    if "datadate" in df.columns:
        raw = df["datadate"]
        as_str = raw.astype(str).str.strip()
        as_str = as_str.str.replace(r"\\.0$", "", regex=True)

        # Prefer strict formats first.
        if as_str.str.contains("-").any():
            parsed = pd.to_datetime(as_str, format="%Y-%m-%d", errors="coerce")
        else:
            parsed = pd.to_datetime(as_str, format="%Y%m%d", errors="coerce")

        # Fallback for unexpected formats.
        if parsed.isna().any():
            fallback = pd.to_datetime(as_str, errors="coerce")
            parsed = parsed.fillna(fallback)

        if parsed.isna().any():
            bad_examples = as_str[parsed.isna()].head(5).tolist()
            raise ValueError(
                "Could not parse some datadate values. Examples: " + ", ".join(map(str, bad_examples))
            )

        df["datadate"] = parsed.dt.strftime("%Y%m%d").astype(int)

    df["ajexdi"] = 1.0

    # Keep the same style as the input (which appears to have an index column).
    df.to_csv(output_path, index=True)
    print(f"Wrote {len(df):,} rows to {output_path}")


if __name__ == "__main__":
    main()
