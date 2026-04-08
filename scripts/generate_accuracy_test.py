#!/usr/bin/env python3
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0.

"""Generate a registered unittest file that runs ``sglang.test.run_eval`` accuracy checks.

Example (DeepSeek-V3–style GSM8K, completion API, CI-sized checkpoint)::

    python scripts/generate_accuracy_test.py \\
      --model lmsys/sglang-ci-dsv3-test \\
      --output test/registered/mla/test_accuracy_deepseek_v3_custom.py \\
      --preset deepseek-v3-gsm8k

Example (custom eval and server args)::

    python scripts/generate_accuracy_test.py \\
      --model moonshotai/Kimi-K2-Thinking \\
      --output test/registered/eval/test_kimi_accuracy_generated.py \\
      --eval-name gsm8k --api chat --min-score 0.72 \\
      --thinking-mode kimi-k2 --temperature 1.0 --top-p 1.0 \\
      --max-tokens 4096 --num-examples 100 \\
      --extra-server-arg --trust-remote-code \\
      --extra-server-arg --tp --extra-server-arg 8 \\
      --cuda-est-time 7200 --cuda-suite nightly-8-gpu-h200 --nightly

Valid ``--eval-name`` values match ``sglang.test.run_eval`` (e.g. gsm8k, mmlu, gpqa, mgsm_en).

The script only prints/writes Python; run pytest/unittest locally after reviewing thresholds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def _slug(s: str) -> str:
    x = re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")
    return x or "model"


def _indent(lines: list[str], spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + ln if ln else "" for ln in lines)


def _build_eval_namespace(
    *,
    eval_name: str,
    api: str,
    max_tokens: int,
    num_examples: int | None,
    num_threads: int,
    thinking_mode: str | None,
    temperature: float | None,
    top_p: float | None,
) -> str:
    parts = [
        "base_url=self.base_url",
        "model=self.model",
        f'eval_name="{eval_name}"',
        f'api="{api}"',
        f"max_tokens={max_tokens}",
        f"num_threads={num_threads}",
    ]
    if num_examples is not None:
        parts.append(f"num_examples={num_examples}")
    if thinking_mode:
        parts.append(f'thinking_mode="{thinking_mode}"')
    if temperature is not None:
        parts.append(f"temperature={temperature!r}")
    if top_p is not None:
        parts.append(f"top_p={top_p!r}")
    inner = ",\n            ".join(parts)
    return f"SimpleNamespace(\n            {inner},\n        )"


def _format_extra_server_args(args_list: list[str]) -> str:
    if not args_list:
        return ""
    # Represent as Python list of strings
    return repr(args_list)


def generate(
    *,
    model: str,
    class_name: str,
    test_method_name: str,
    docstring: str,
    eval_name: str,
    api: str,
    min_score: float,
    max_tokens: int,
    num_examples: int | None,
    num_threads: int,
    thinking_mode: str | None,
    temperature: float | None,
    top_p: float | None,
    extra_server_args: list[str],
    enable_torch_compile_on_cuda: bool,
    cuda_graph_max_bs: int | None,
    chunked_prefill_size: int | None,
    trust_remote_code: bool,
    register_cuda: bool,
    cuda_est_time: float,
    cuda_suite: str,
    cuda_nightly: bool,
    register_amd: bool,
    amd_est_time: float,
    amd_suite: str,
    amd_disabled: str | None,
    launch_timeout_multiplier: float,
) -> str:
    eval_ns = _build_eval_namespace(
        eval_name=eval_name,
        api=api,
        max_tokens=max_tokens,
        num_examples=num_examples,
        num_threads=num_threads,
        thinking_mode=thinking_mode,
        temperature=temperature,
        top_p=top_p,
    )

    base_other: list[str] = []
    if trust_remote_code:
        base_other.append("--trust-remote-code")
    if chunked_prefill_size is not None:
        base_other.extend(["--chunked-prefill-size", str(chunked_prefill_size)])

    # Combine: base first, then user extras (user can override ordering by passing full list only — keep simple)
    combined = base_other + extra_server_args

    other_args_repr = _format_extra_server_args(combined)

    register_lines: list[str] = []
    if register_cuda:
        nightly_kw = ", nightly=True" if cuda_nightly else ""
        register_lines.append(
            f'register_cuda_ci(est_time={cuda_est_time!r}, suite="{cuda_suite}"{nightly_kw})'
        )
    if register_amd:
        if amd_disabled:
            register_lines.append(
                f'register_amd_ci(est_time={amd_est_time!r}, suite="{amd_suite}", disabled="{amd_disabled}")'
            )
        else:
            register_lines.append(
                f'register_amd_ci(est_time={amd_est_time!r}, suite="{amd_suite}")'
            )

    register_block = "\n".join(register_lines) if register_lines else "# register_*_ci(...)  # add CI registration before merging"

    ci_imports = "from sglang.test.ci.ci_register import "
    ci_parts = []
    if register_cuda:
        ci_parts.append("register_cuda_ci")
    if register_amd:
        ci_parts.append("register_amd_ci")
    if ci_parts:
        ci_import_line = ci_imports + ", ".join(ci_parts)
    else:
        ci_import_line = "# from sglang.test.ci.ci_register import register_cuda_ci  # enable --register-amd or drop --no-register-cuda"

    torch_compile_block = ""
    if enable_torch_compile_on_cuda:
        cg = (
            f'["--enable-torch-compile", "--cuda-graph-max-bs", "{cuda_graph_max_bs}"]'
            if cuda_graph_max_bs is not None
            else '["--enable-torch-compile"]'
        )
        # Must stay indented inside setUpClass (do not use dedent — it breaks class-body indent).
        torch_compile_block = (
            f"        if is_cuda():\n            other_args.extend({cg})\n"
        )

    timeout_expr = (
        f"DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * {launch_timeout_multiplier!r}"
        if launch_timeout_multiplier != 1.0
        else "DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH"
    )

    docstring_py = docstring.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    src = f'''# Generated by scripts/generate_accuracy_test.py — review before committing.
# Model: {model}

import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_cuda, is_hip, kill_process_tree
{ci_import_line}
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

{register_block}


class {class_name}(CustomTestCase):
    """{docstring_py}"""

    @classmethod
    def setUpClass(cls):
        cls.model = "{model}"
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = {other_args_repr or "[]"}
{torch_compile_block}
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout={timeout_expr},
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @unittest.skipUnless(is_cuda(), "CUDA required for this accuracy test")
    @unittest.skipIf(is_hip(), "Adjust baseline on AMD before enabling")
    def {test_method_name}(self):
        args = {eval_ns}
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], {min_score!r})


if __name__ == "__main__":
    unittest.main()
'''
    return src


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write the generated test file (e.g. test/registered/mla/test_foo.py)",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="Test class name (default: derived from --model)",
    )
    parser.add_argument(
        "--test-method",
        default="test_accuracy",
        help="Test method name (default: test_accuracy)",
    )
    parser.add_argument(
        "--description",
        default="",
        help="One-line docstring for the test class",
    )
    parser.add_argument("--eval-name", default="gsm8k", help="run_eval eval_name")
    parser.add_argument("--api", default="completion", choices=["chat", "completion"])
    parser.add_argument("--min-score", type=float, default=0.62, help="assert metrics['score'] > this")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--num-threads", type=int, default=128)
    parser.add_argument("--thinking-mode", default=None, help="e.g. kimi-k2, deepseek-v3")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--extra-server-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Repeat for each server CLI token (e.g. --extra-server-arg --tp --extra-server-arg 8)",
    )
    parser.add_argument(
        "--no-torch-compile",
        action="store_true",
        help="Do not append --enable-torch-compile on CUDA",
    )
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        default=2,
        help="With torch compile on CUDA, set --cuda-graph-max-bs (default: 2)",
    )
    parser.add_argument(
        "--chunked-prefill-size",
        type=int,
        default=256,
        help="Set --chunked-prefill-size (default 256); use --chunked-prefill-size -1 to omit",
    )
    parser.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Omit --trust-remote-code",
    )
    parser.add_argument("--no-register-cuda", action="store_true")
    parser.add_argument("--cuda-est-time", type=float, default=600.0)
    parser.add_argument("--cuda-suite", default="stage-b-test-1-gpu-large")
    parser.add_argument("--cuda-nightly", action="store_true")
    parser.add_argument("--register-amd", action="store_true")
    parser.add_argument("--amd-est-time", type=float, default=300.0)
    parser.add_argument("--amd-suite", default="stage-b-test-1-gpu-small-amd")
    parser.add_argument("--amd-disabled", default=None, help="If set, register_amd_ci(..., disabled=...)")
    parser.add_argument(
        "--launch-timeout-multiplier",
        type=float,
        default=1.0,
        help="Multiply DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH for slow JIT / large models",
    )
    parser.add_argument(
        "--preset",
        choices=["none", "deepseek-v3-gsm8k"],
        default="none",
        help="deepseek-v3-gsm8k: CI DSV3-style gsm8k + completion + torch compile + chunked prefill",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout only")

    args = parser.parse_args(argv)

    model = args.model
    class_name = args.class_name or f"TestAccuracy{_slug(model)}"
    test_method = args.test_method if args.test_method.startswith("test_") else f"test_{args.test_method}"

    eval_name = args.eval_name
    api = args.api
    min_score = args.min_score
    max_tokens = args.max_tokens
    num_examples = args.num_examples
    num_threads = args.num_threads
    thinking_mode = args.thinking_mode
    temperature = args.temperature
    top_p = args.top_p
    extra_server_args = list(args.extra_server_arg)

    enable_torch = not args.no_torch_compile
    chunked = args.chunked_prefill_size
    if chunked is not None and chunked < 0:
        chunked = None
    trust_remote = not args.no_trust_remote_code

    register_cuda = not args.no_register_cuda
    cuda_nightly = args.cuda_nightly

    if args.preset == "deepseek-v3-gsm8k":
        eval_name = "gsm8k"
        api = "completion"
        min_score = 0.62
        max_tokens = 512
        num_examples = 200
        num_threads = 128
        thinking_mode = None
        temperature = None
        top_p = None
        enable_torch = True
        chunked = 256
        trust_remote = True
        if not args.extra_server_arg:
            extra_server_args = []

    desc = args.description or f"Accuracy regression for {model} ({eval_name})."

    text = generate(
        model=model,
        class_name=class_name,
        test_method_name=test_method,
        docstring=desc,
        eval_name=eval_name,
        api=api,
        min_score=min_score,
        max_tokens=max_tokens,
        num_examples=num_examples,
        num_threads=num_threads,
        thinking_mode=thinking_mode,
        temperature=temperature,
        top_p=top_p,
        extra_server_args=extra_server_args,
        enable_torch_compile_on_cuda=enable_torch,
        cuda_graph_max_bs=args.cuda_graph_max_bs,
        chunked_prefill_size=chunked,
        trust_remote_code=trust_remote,
        register_cuda=register_cuda,
        cuda_est_time=args.cuda_est_time,
        cuda_suite=args.cuda_suite,
        cuda_nightly=cuda_nightly,
        register_amd=args.register_amd,
        amd_est_time=args.amd_est_time,
        amd_suite=args.amd_suite,
        amd_disabled=args.amd_disabled,
        launch_timeout_multiplier=args.launch_timeout_multiplier,
    )

    out = Path(args.output)
    if args.dry_run:
        print(text)
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
