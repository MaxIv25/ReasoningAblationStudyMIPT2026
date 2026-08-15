from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_tilelang_temp_cleanup_is_configured_before_torch_import():
    for filename in ("train_grpo.py", "train_prime.py"):
        source = (ROOT / "src" / filename).read_text()
        cleanup = 'os.environ.setdefault("TILELANG_CLEANUP_TEMP_FILES", "1")'
        assert cleanup in source
        assert source.index(cleanup) < source.index("import torch")
