#!/usr/bin/env python3
"""
Summarize MIGraphX ONNX Model Zoo test logs.

Reads *.log files from fp32/ and fp16/ subfolders of a results directory,
counts passes/failures, extracts a short failure message, and writes:
  - summary.json
  - summary.md

Usage:
  python3 scripts/summarize_onnx_logs.py --results <RESULTS_DIR>
  # optional overrides
  python3 scripts/summarize_onnx_logs.py --results <DIR> \
      --out-json <PATH> --out-md <PATH>

Exit codes:
  0 on success (even if there are failures in tests)
  2 if results directory cannot be read
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path

FAIL_PAT = re.compile(
    r"(Traceback \(most recent call last\)|\bERROR\b|AssertionError|Segmentation fault|^error:)",
    re.I | re.M,
)
TBLOCK_PAT = re.compile(r"Traceback \(most recent call last\):([\s\S]*?)(?:\n\s*\n|\Z)")


def looks_failed(text: str) -> bool:
    return bool(FAIL_PAT.search(text))


def failure_message(text: str) -> str:
    # last traceback block, last non-empty line
    blocks = list(TBLOCK_PAT.finditer(text))
    if blocks:
        for line in reversed(blocks[-1].group(1).strip().splitlines()):
            line = line.strip()
            if line:
                return line
    # fallback: last interesting line
    for line in reversed([l.strip() for l in text.splitlines() if l.strip()]):
        if re.search(r"(error|exception|failed|segmentation fault|assert)", line, re.I):
            return line
    return "failed (see log)"


def summarize(results_dir: Path) -> dict:
    precs = ("fp32", "fp16")
    summary = {"totals": {"pass": 0, "fail": 0}, "regressions": {}}

    for prec in precs:
        d = results_dir / prec
        reg = {"passed": [], "failed": []}
        if d.is_dir():
            files = sorted(p for p in d.glob("*.log"))
            for p in files:
                model = p.stem
                try:
                    txt = p.read_text(errors="ignore")
                except Exception:
                    txt = ""
                if looks_failed(txt):
                    reg["failed"].append({"model": model, "message": failure_message(txt)})
                    summary["totals"]["fail"] += 1
                else:
                    reg["passed"].append(model)
                    summary["totals"]["pass"] += 1
        summary["regressions"][prec] = reg
    return summary


def write_outputs(summary: dict, out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, indent=2))

    lines = [
        "## Totals",
        f"- PASS: {summary['totals']['pass']}",
        f"- FAIL: {summary['totals']['fail']}",
        "",
    ]
    for prec in ("fp32", "fp16"):
        reg = summary["regressions"][prec]
        lines.append(f"## {prec.upper()}")
        lines.append(f"**Passed ({len(reg['passed'])})**")
        if reg["passed"]:
            lines.extend([f"- {m}" for m in reg["passed"]])
        else:
            lines.append("- none")
        lines.append("")
        lines.append(f"**Failed ({len(reg['failed'])})**")
        if reg["failed"]:
            lines.extend([f"- {it['model']}: `{it['message']}`" for it in reg["failed"]])
        else:
            lines.append("- none")
        lines.append("")
    out_md.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Results directory containing fp32/fp16 log folders")
    ap.add_argument("--out-json", default=None, help="Path to write summary.json (default: <results>/summary.json)")
    ap.add_argument("--out-md", default=None, help="Path to write summary.md (default: <results>/summary.md)")
    args = ap.parse_args()

    results_dir = Path(args.results).expanduser().resolve()
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}")
        return 2

    summary = summarize(results_dir)

    out_json = Path(args.out_json) if args.out_json else results_dir / "summary.json"
    out_md = Path(args.out_md) if args.out_md else results_dir / "summary.md"
    write_outputs(summary, out_json, out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
