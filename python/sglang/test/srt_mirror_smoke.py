# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Helpers for auto-generated per-module smoke tests under ``test/registered/unit/srt/``."""

from __future__ import annotations

import ast
import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = [
    "Symbol",
    "assert_symbol_resolves",
    "collect_ast_symbols",
    "module_name_from_srt_relative_path",
    "parse_source_file",
    "read_module_source_path",
]


@dataclass(frozen=True)
class Symbol:
    kind: str
    name: str
    class_name: Optional[str] = None


class _CollectFunctions(ast.NodeVisitor):
    def __init__(self) -> None:
        self.symbols: List[Symbol] = []
        self._class_stack: List[str] = []
        self._function_depth = 0

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_functionlike(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_functionlike(node, is_async=True)

    def _visit_functionlike(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, *, is_async: bool
    ) -> None:
        if self._class_stack:
            if self._function_depth == 0:
                self.symbols.append(
                    Symbol(
                        "async_method" if is_async else "method",
                        node.name,
                        class_name=self._class_stack[-1],
                    )
                )
        else:
            if self._function_depth == 0:
                self.symbols.append(
                    Symbol("async_fn" if is_async else "fn", node.name)
                )

        self._function_depth += 1
        self.generic_visit(node)
        self._function_depth -= 1


def collect_ast_symbols(source: str) -> List[Symbol]:
    tree = ast.parse(source)
    visitor = _CollectFunctions()
    visitor.visit(tree)
    return visitor.symbols


def parse_source_file(path: Path) -> Tuple[str, List[Symbol]]:
    text = path.read_text(encoding="utf-8")
    ast.parse(text, filename=str(path))
    return text, collect_ast_symbols(text)


def module_name_from_srt_relative_path(rel: Path) -> str:
    """Map ``Path('utils/common.py')`` relative to ``sglang/srt`` to import name."""
    parts = rel.with_suffix("").parts
    if not parts:
        raise ValueError(f"empty relative path: {rel}")
    if parts[-1] == "__init__":
        inner = parts[:-1]
        if not inner:
            return "sglang.srt"
        return "sglang.srt." + ".".join(inner)
    return "sglang.srt." + ".".join(parts)


def read_module_source_path(module_name: str) -> Optional[Path]:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return None
    origin = spec.origin
    if origin in ("built-in", "frozen"):
        return None
    path = Path(origin)
    if path.suffix != ".py":
        return None
    return path


def assert_symbol_resolves(module_name: str, mod: object, sym: Symbol) -> None:
    if sym.class_name is None:
        assert hasattr(mod, sym.name), (
            f"{module_name}: missing module-level symbol {sym.name!r}"
        )
        return
    assert hasattr(mod, sym.class_name), (
        f"{module_name}: missing class {sym.class_name!r} for member {sym.name!r}"
    )
    cls = getattr(mod, sym.class_name)
    assert hasattr(cls, sym.name), (
        f"{module_name}: class {sym.class_name!r} missing member {sym.name!r}"
    )
