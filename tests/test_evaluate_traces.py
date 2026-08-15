import gzip
import json
import sys
from types import SimpleNamespace

import src.evaluate as evaluate


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeLLM:
    init_kwargs = None
    sampling_kwargs = None

    def __init__(self, **kwargs):
        self.__class__.init_kwargs = kwargs

    def generate(self, *, prompts, sampling_params, **kwargs):
        self.__class__.sampling_kwargs = sampling_params.kwargs
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="reasoning \\boxed{2}",
                        token_ids=[1, 2, 3],
                        finish_reason="stop",
                    ),
                    SimpleNamespace(
                        text="other \\boxed{2}",
                        token_ids=[4, 5],
                        finish_reason="stop",
                    ),
                ]
            )
            for _ in prompts
        ]


def test_evaluate_writes_every_trace_and_locks_seed(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=_FakeLLM, SamplingParams=_FakeSamplingParams),
    )
    monkeypatch.setattr(evaluate, "patch_vllm_language_model_only", lambda *a, **k: None)
    monkeypatch.setattr(
        evaluate,
        "load_gsm8k_test",
        lambda: [
            {"question": "one plus one?", "answer": "2", "source": "gsm8k"},
            {"question": "again?", "answer": "2", "source": "gsm8k"},
        ],
    )
    monkeypatch.setattr(evaluate.torch.cuda, "empty_cache", lambda: None)

    traces = tmp_path / "traces.jsonl.gz"
    results = evaluate.evaluate_model(
        "fake-model",
        benchmarks=["gsm8k"],
        num_samples=2,
        max_new_tokens=32,
        seed=17,
        traces_output=str(traces),
    )

    with gzip.open(traces, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]

    assert len(rows) == 4
    assert rows[0]["generated_text"] == "reasoning \\boxed{2}"
    assert rows[0]["response_tokens"] == 3
    assert results["gsm8k"]["accuracy"] == 100.0
    assert _FakeLLM.init_kwargs["seed"] == 17
    assert _FakeLLM.init_kwargs["language_model_only"] is True
    assert _FakeLLM.sampling_kwargs["seed"] == 17
