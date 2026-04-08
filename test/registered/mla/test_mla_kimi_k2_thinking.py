# Accuracy regression for moonshotai/Kimi-K2-Thinking (MoE + MLA, native INT4).
# Model card: https://huggingface.co/moonshotai/Kimi-K2-Thinking
#
# Mirrors the GSM8K eval flow in test_mla_deepseek_v3.py, but uses the chat API with
# thinking_mode="kimi-k2" (see sglang.test.run_eval THINKING_MODE_CHOICES) and server
# flags --reasoning-parser / --tool-call-parser consistent with other Kimi-K2 tests.
#
# MMLU: uses sglang.test.run_eval eval_name="mmlu" (simple-evals MMLU). The model card
# reports MMLU-Pro; this harness tracks classic MMLU as a CI-friendly knowledge check.

import unittest
from types import SimpleNamespace

from sglang.srt.utils import is_cuda, is_hip, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# Large MoE: multi-GPU, long runtime. Align with other Kimi-K2.5 / stress registrations.
register_cuda_ci(
    est_time=10800,
    suite="nightly-8-gpu-h200",
    nightly=True,
)

MODEL_ID = "moonshotai/Kimi-K2-Thinking"

_COMMON_EVAL_KWARGS = dict(
    api="chat",
    thinking_mode="kimi-k2",
    temperature=1.0,
    top_p=1.0,
)


class TestMLAKimiK2ThinkingAccuracy(CustomTestCase):
    """GSM8K + MMLU accuracy for Kimi-K2-Thinking with Kimi thinking + parsers.

    Recommended decoding settings from the model card (temperature ~1.0) are reflected
    in the eval args. Requires sufficient GPU memory and TP (default tp=8).
    One server process serves both tests.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL_ID
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--mem-fraction-static",
            "0.8",
            "--reasoning-parser",
            "kimi_k2",
            "--tool-call-parser",
            "kimi_k2",
            "--chunked-prefill-size",
            "256",
        ]
        if is_cuda():
            other_args.extend(["--enable-torch-compile", "--cuda-graph-max-bs", "2"])
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 2,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    @unittest.skipUnless(is_cuda(), "Kimi-K2-Thinking INT4 path is exercised on CUDA")
    @unittest.skipIf(is_hip(), "Kimi-K2-Thinking accuracy baseline is tracked on NVIDIA")
    def test_gsm8k_accuracy(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=4096,
            num_examples=200,
            num_threads=128,
            **_COMMON_EVAL_KWARGS,
        )
        metrics = run_eval(args)
        print(metrics)

        # GSM8K is a coarse sanity check; tune if eval/sampling pipeline changes.
        self.assertGreater(metrics["score"], 0.72)

    @unittest.skipUnless(is_cuda(), "Kimi-K2-Thinking INT4 path is exercised on CUDA")
    @unittest.skipIf(is_hip(), "Kimi-K2-Thinking accuracy baseline is tracked on NVIDIA")
    def test_mmlu_accuracy(self):
        """Classic MMLU subset (simple-evals); model card MMLU-Pro 84.6 is a distinct bench."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            max_tokens=2048,
            num_examples=500,
            num_threads=128,
            **_COMMON_EVAL_KWARGS,
        )
        metrics = run_eval(args)
        print(metrics)

        self.assertGreater(metrics["score"], 0.78)


if __name__ == "__main__":
    unittest.main()
