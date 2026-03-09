import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semtero import reranker as reranker_module
from semtero.reranker import Reranker


GB = 1024**3


def test_select_device_prefers_gpu_with_enough_total_and_free_vram(monkeypatch):
    mem_info = {
        0: (9 * GB, 12 * GB),
        1: (11 * GB, 16 * GB),
    }

    monkeypatch.setattr(reranker_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(reranker_module.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        reranker_module.Reranker,
        "_get_device_memory_info",
        lambda self, idx: mem_info[idx],
    )

    reranker = Reranker(min_gpu_vram_gb=8.0)

    assert reranker.device.type == "cuda"
    assert reranker.device.index == 1


def test_select_device_falls_back_to_cpu_when_free_vram_is_below_threshold(monkeypatch):
    monkeypatch.setattr(reranker_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(reranker_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        reranker_module.Reranker,
        "_get_device_memory_info",
        lambda self, idx: (6 * GB, 24 * GB),
    )

    reranker = Reranker(min_gpu_vram_gb=8.0)

    assert reranker.device.type == "cpu"


def test_rerank_skips_loading_model_for_empty_results(monkeypatch):
    tokenizer_calls = []
    model_calls = []

    monkeypatch.setattr(
        reranker_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        reranker_module.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: model_calls.append((args, kwargs)),
    )

    reranker = Reranker()

    assert reranker.rerank([], "query") == []
    assert tokenizer_calls == []
    assert model_calls == []


def test_release_device_moves_model_off_gpu_and_clears_cuda_cache(monkeypatch):
    calls = []

    class FakeModel:
        def to(self, device):
            calls.append(("to", str(device)))
            return self

    monkeypatch.setattr(reranker_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        reranker_module.torch.cuda,
        "empty_cache",
        lambda: calls.append(("empty_cache",)),
    )
    monkeypatch.setattr(
        reranker_module.torch.cuda,
        "ipc_collect",
        lambda: calls.append(("ipc_collect",)),
    )

    reranker = Reranker()
    reranker.device = reranker_module.torch.device("cuda:0")
    reranker.model = FakeModel()

    reranker.release_device()

    assert reranker.model is None
    assert calls == [("to", "cpu"), ("empty_cache",), ("ipc_collect",)]
