import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_get_summary_writer_skips_missing_tensorboard(monkeypatch, tmp_path):
    import scripts.finetuning as finetuning

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name):
        if name == "tensorboard":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    writer = finetuning.get_summary_writer(str(tmp_path / "tensorboard"))
    assert writer is None
