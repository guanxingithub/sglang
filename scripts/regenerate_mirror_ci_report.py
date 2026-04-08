#!/usr/bin/env python3
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Regenerate ``mirror_ci_test_index.csv`` and ``MIRROR_CI_TEST_REPORT.md`` for SRT / sgl_kernel mirror CI tests.

Does not require a full ``pip install`` of ``sglang`` (loads mirror smoke helpers by file path).

Usage::

    ./scripts/regenerate_mirror_ci_report.sh
    python3 scripts/regenerate_mirror_ci_report.py
    python3 scripts/regenerate_mirror_ci_report.py --csv-only
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib.util
import re
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def sym_str(s) -> str:
    if s.class_name:
        return f"{s.class_name}.{s.name}"
    return s.name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-only", action="store_true", help="Only write mirror_ci_test_index.csv")
    ap.add_argument("--md-only", action="store_true", help="Only rewrite MIRROR_CI_TEST_REPORT.md (requires existing CSV)")
    args = ap.parse_args()
    if args.csv_only and args.md_only:
        print("error: choose at most one of --csv-only and --md-only", file=sys.stderr)
        return 2

    repo = _repo_root()
    out_csv = repo / "test" / "registered" / "unit" / "mirror_ci_test_index.csv"
    out_md = repo / "test" / "registered" / "unit" / "MIRROR_CI_TEST_REPORT.md"

    _srt = _load(repo / "python" / "sglang" / "test" / "srt_mirror_smoke.py", "_srt_mirror_smoke")
    _sgk = _load(repo / "python" / "sglang" / "test" / "sgl_kernel_mirror_smoke.py", "_sgk_mirror_smoke")

    srt_re = re.compile(r"RELATIVE_SRT_PATH\s*=\s*Path\((['\"])(.+?)\1\)")
    sgk_re = re.compile(r"RELATIVE_PKG_PATH\s*=\s*Path\((['\"])(.+?)\1\)")

    rows: list[dict] = []

    def parse_source_bom_ok(src: Path, suite: str):
        text = src.read_text(encoding="utf-8-sig")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(text, filename=str(src))
            if suite == "srt":
                syms = _srt.collect_ast_symbols(text)
            else:
                syms = _sgk.collect_ast_symbols(text)
        return syms

    def scan_tree(test_root: Path, kind: str):
        if not test_root.is_dir():
            return
        for test_py in sorted(test_root.rglob("test_*.py")):
            text = test_py.read_text(encoding="utf-8", errors="replace")
            if kind == "srt":
                m = srt_re.search(text)
                if not m:
                    continue
                rel = Path(m.group(2))
                src = repo / "python" / "sglang" / "srt" / rel
                mod = _srt.module_name_from_srt_relative_path(rel)
                syms = parse_source_bom_ok(src, "srt")
            else:
                m = sgk_re.search(text)
                if not m:
                    continue
                rel = Path(m.group(2))
                src = repo / "sgl-kernel" / "python" / "sgl_kernel" / rel
                mod = _sgk.module_name_from_sgl_kernel_relative_path(rel)
                syms = parse_source_bom_ok(src, "sgl_kernel")
            rel_test = test_py.relative_to(repo)
            rel_src = src.relative_to(repo)
            rows.append(
                {
                    "suite": kind,
                    "unittest_file": str(rel_test).replace("\\", "/"),
                    "source_file": str(rel_src).replace("\\", "/"),
                    "python_module": mod,
                    "unittest_methods": "test_source_parses;test_symbols_when_import_succeeds",
                    "symbol_count": len(syms),
                    "symbols": ";".join(sym_str(s) for s in syms),
                }
            )

    if not args.md_only:
        scan_tree(repo / "test" / "registered" / "unit" / "srt", "srt")
        scan_tree(repo / "test" / "registered" / "unit" / "sgl_kernel", "sgl_kernel")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "suite",
                    "unittest_file",
                    "source_file",
                    "python_module",
                    "unittest_methods",
                    "symbol_count",
                    "symbols",
                ],
            )
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv.relative_to(repo)} ({len(rows)} rows)", flush=True)

    if args.csv_only:
        return 0

    if args.md_only:
        if not out_csv.is_file():
            print(f"error: {out_csv} not found; run without --md-only first", file=sys.stderr)
            return 1
        rows = []
        with out_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    srt_rows = [r for r in rows if r["suite"] == "srt"]
    sgk_rows = [r for r in rows if r["suite"] == "sgl_kernel"]

    def sym_sum(rs):
        return sum(int(r["symbol_count"]) for r in rs)

    n_srt = len(srt_rows)
    n_sgk = len(sgk_rows)
    sum_srt = sym_sum(srt_rows)
    sum_sgk = sym_sum(sgk_rows)
    total_files = n_srt + n_sgk
    total_syms = sum_srt + sum_sgk

    by_seg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in srt_rows:
        rel = Path(r["source_file"]).relative_to("python/sglang/srt")
        seg = rel.parts[0] if rel.parts else "."
        by_seg[seg][0] += 1
        by_seg[seg][1] += int(r["symbol_count"])

    top_segments = sorted(by_seg.items(), key=lambda x: -x[1][0])[:10]

    def pick_example(kind: str, needle: str) -> dict | None:
        for r in rows:
            if r["suite"] != kind:
                continue
            if needle in r["unittest_file"]:
                return r
        return None

    ex_srt = pick_example("srt", "batch_invariant_ops/test_batch_invariant_ops.py")
    if not ex_srt and srt_rows:
        ex_srt = max(srt_rows, key=lambda r: int(r["symbol_count"]))
    ex_sgk = pick_example("sgl_kernel", "test_allreduce.py")
    if not ex_sgk and sgk_rows:
        ex_sgk = sgk_rows[0]

    gen_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    seg_rows_md = "\n".join(
        f"| `{name}` | {counts[0]} | {counts[1]:,} |" for name, counts in top_segments
    )

    ex_srt_md = (
        f"| `{ex_srt['unittest_file']}` | `{ex_srt['source_file']}` | `{ex_srt['python_module']}` |"
        if ex_srt
        else "| *(no SRT mirror rows)* | | |"
    )
    ex_sgk_md = (
        f"| `{ex_sgk['unittest_file']}` | `{ex_sgk['source_file']}` | `{ex_sgk['python_module']}` |"
        if ex_sgk
        else "| *(no sgl_kernel mirror rows)* | | |"
    )

    md = f"""# Mirror unittest CI tests: SRT and `sgl_kernel`

**Scope:** Auto-generated **CPU mirror smoke tests** under `test/registered/unit/srt/` and `test/registered/unit/sgl_kernel/`. Each file pairs **one** Python module under `python/sglang/srt/` or `sgl-kernel/python/sgl_kernel/` with **one** unittest module.

**Machine-readable index:** `mirror_ci_test_index.csv` (same directory) — one row per mirror test, with full symbol lists.

**Generated:** {gen_date} (UTC; counts from `scripts/regenerate_mirror_ci_report.py`).

---

## CI registration

Every mirror file calls:

```python
register_cpu_ci(est_time=1, suite="stage-a-test-cpu")
```

(`test/run_suite.py` discovers these under `test/registered/`.)

---

## Unittest surface (same for every mirror file)

| Item | Value |
|------|--------|
| **Test class** | `TestMirrorSmoke` |
| **`test_source_parses`** | AST-parse the mirrored **source file** (syntax smoke). |
| **`test_symbols_when_import_succeeds`** | Collect **symbols** from the source AST (see below); if `import <module>` succeeds, assert each symbol **resolves** on the imported module (`hasattr` for module-level names; class + attribute for methods). If import fails, the test **skips** (optional dependency). |

Helpers live in:

- `python/sglang/test/srt_mirror_smoke.py` — SRT
- `python/sglang/test/sgl_kernel_mirror_smoke.py` — `sgl_kernel`

---

## Which “Python functions” are tracked

The mirror does **not** enumerate every callable in the runtime; it uses a **static AST walk** (`collect_ast_symbols`):

- **Module level:** each top-level `def` / `async def` (not nested inside another function).
- **Inside classes:** each **direct** method (`def` / `async def` immediately under `class`, not nested inside another function in the class body).

Nested functions, lambdas, and non-function definitions are **not** listed as separate symbols.

Symbols are checked with `assert_symbol_resolves` (module attribute or `Class.method`).

---

## Path correspondence

### SRT (`suite=srt`)

| Role | Path pattern |
|------|----------------|
| Unittest file | `test/registered/unit/srt/<relative>/test_<stem>.py` |
| Source file | `python/sglang/srt/<relative>/<stem>.py` |
| Import name | `sglang.srt.<relative dotted>.<stem>` (see `module_name_from_srt_relative_path`; `__init__.py` maps to package). |

The generated test embeds `RELATIVE_SRT_PATH = Path('...')` pointing at the file under `python/sglang/srt/`.

### `sgl_kernel` (`suite=sgl_kernel`)

| Role | Path pattern |
|------|----------------|
| Unittest file | `test/registered/unit/sgl_kernel/<relative>/test_<stem>.py` |
| Source file | `sgl-kernel/python/sgl_kernel/<relative>/<stem>.py` |
| Import name | `sgl_kernel.<relative dotted>.<stem>`. |

The generated test embeds `RELATIVE_PKG_PATH = Path('...')` under `sgl-kernel/python/sgl_kernel/`.

---

## Summary counts (this tree)

| Suite | Mirror unittest files | AST symbol entries (sum of `symbol_count`) |
|--------|-------------------------|---------------------------------------------|
| **srt** | {n_srt} | {sum_srt:,} |
| **sgl_kernel** | {n_sgk} | {sum_sgk:,} |
| **Total** | **{total_files}** | **{total_syms:,}** |

### SRT: top subdirectories under `python/sglang/srt/` (by number of mirror files)

| Subdirectory | Mirror files | Symbol entries |
|--------------|-------------:|-----------------:|
{seg_rows_md}

(Full per-segment breakdown is derivable from the CSV.)

---

## Example rows (abbreviated)

**SRT — example**

| Unittest | Source | Python module |
|----------|--------|----------------|
{ex_srt_md}

**`sgl_kernel` — example**

| Unittest | Source | Python module |
|----------|--------|----------------|
{ex_sgk_md}

See the `symbols` column in `mirror_ci_test_index.csv` for the full `;`-separated list per file (e.g. `ClassName.method` or bare `function_name`).

---

## Regenerating mirrors and this report

```bash
# From repo root — regenerate mirror unittest files from sources
python3 scripts/generate_srt_mirror_unit_tests.py --clean
python3 scripts/generate_sgl_kernel_mirror_unit_tests.py --clean

# Refresh CSV + this markdown (no sglang pip install required)
./scripts/regenerate_mirror_ci_report.sh
```

Column definitions for `mirror_ci_test_index.csv`:

| Column | Meaning |
|--------|---------|
| `suite` | `srt` or `sgl_kernel` |
| `unittest_file` | Path to generated `test_*.py` |
| `source_file` | Mirrored `.py` under SRT or `sgl_kernel` |
| `python_module` | Import string used when resolving symbols |
| `unittest_methods` | Always `test_source_parses;test_symbols_when_import_succeeds` |
| `symbol_count` | Number of AST-tracked symbols |
| `symbols` | `;`-separated list (`Class.method` or `fn`) |
"""
    out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {out_md.relative_to(repo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
