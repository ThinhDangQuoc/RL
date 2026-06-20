#!/usr/bin/env python3
"""Convert JSON to CSV.

Supports:
- JSON array of objects (most common)
- JSON object (single row)
- JSON Lines / NDJSON (one JSON object per line)

Example:
  python convert_csv.py ACB_2.json ACB_2.csv

If output path is omitted, it uses the same name with .csv extension.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


JsonValue = Any


@dataclass(frozen=True)
class ConvertOptions:
    input_path: Path
    output_path: Path
    delimiter: str
    encoding: str
    flatten: bool
    flatten_sep: str


def _flatten_dict(d: dict[str, Any], *, sep: str) -> dict[str, Any]:
    """Flatten nested dicts into dot-keys: {'a': {'b': 1}} -> {'a.b': 1}."""
    out: dict[str, Any] = {}
    stack: list[tuple[str, Any]] = [("", d)]

    while stack:
        prefix, cur = stack.pop()
        if isinstance(cur, dict):
            for k, v in cur.items():
                key = f"{prefix}{sep}{k}" if prefix else str(k)
                if isinstance(v, dict):
                    stack.append((key, v))
                else:
                    out[key] = v
        else:
            out[prefix] = cur

    return out


def _normalize_cell_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    # For lists/dicts/other objects, store as JSON string to keep information.
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _load_json(path: Path, *, encoding: str) -> JsonValue:
    """Load JSON; if standard JSON parsing fails, try JSON Lines."""
    text = path.read_text(encoding=encoding)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try JSONL/NDJSON
        rows: list[Any] = []
        for idx, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Cannot parse JSON (neither JSON nor JSONL). Error at line {idx}: {exc}"
                ) from exc
        return rows


def _coerce_to_rows(root: JsonValue) -> list[dict[str, Any]]:
    """Coerce root JSON value into list of dict rows."""
    if isinstance(root, list):
        rows: list[dict[str, Any]] = []
        for i, item in enumerate(root):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected list of objects, but element {i} is {type(item).__name__}"
                )
            rows.append(item)
        return rows

    if isinstance(root, dict):
        # If it looks like a wrapper object containing a list of records, pick it.
        for key in ("data", "records", "items", "rows", "result", "results"):
            maybe = root.get(key)
            if isinstance(maybe, list) and all(isinstance(x, dict) for x in maybe):
                return list(maybe)  # type: ignore[arg-type]
        return [root]

    raise ValueError(
        f"Unsupported JSON root type: {type(root).__name__}. Expected object or array."
    )


def _discover_fieldnames(rows: Iterable[dict[str, Any]]) -> list[str]:
    """Stable union of keys in order of first appearance."""
    seen: set[str] = set()
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    return fieldnames


def convert_json_to_csv(options: ConvertOptions) -> None:
    root = _load_json(options.input_path, encoding=options.encoding)
    rows = _coerce_to_rows(root)

    processed_rows: list[dict[str, Any]] = []
    for row in rows:
        if options.flatten:
            processed_rows.append(_flatten_dict(row, sep=options.flatten_sep))
        else:
            processed_rows.append(row)

    fieldnames = _discover_fieldnames(processed_rows)

    options.output_path.parent.mkdir(parents=True, exist_ok=True)

    with options.output_path.open("w", newline="", encoding=options.encoding) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            delimiter=options.delimiter,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in processed_rows:
            writer.writerow({k: _normalize_cell_value(row.get(k)) for k in fieldnames})


def _default_output_path(input_path: Path) -> Path:
    # Keep the name, change extension to .csv
    return input_path.with_suffix(".csv")


def parse_args(argv: list[str] | None = None) -> ConvertOptions:
    parser = argparse.ArgumentParser(description="Convert JSON file to CSV")
    parser.add_argument("--input", help="Path to input .json/.jsonl file")
    parser.add_argument(
        "--output",
        nargs="?",
        default=None,
        help="Path to output .csv file (default: same name with .csv)",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="CSV delimiter (default: ,)",
    )

    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding for reading/writing (default: utf-8)",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Flatten nested JSON objects into dot-keys (default: off)",
    )
    parser.add_argument(
        "--flatten-sep",
        default=".",
        help="Separator for flattened keys (default: .)",
    )

    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else _default_output_path(input_path)

    if len(args.delimiter) != 1:
        raise SystemExit("--delimiter must be a single character")

    return ConvertOptions(
        input_path=input_path,
        output_path=output_path,
        delimiter=args.delimiter,
        encoding=args.encoding,
        flatten=bool(args.flatten),
        flatten_sep=args.flatten_sep,
    )


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    convert_json_to_csv(options)
    print(f"Wrote CSV: {options.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())