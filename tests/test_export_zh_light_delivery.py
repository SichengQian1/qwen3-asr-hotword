import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from qwen_hotword.evaluation import zh_en_delivery as base
from qwen_hotword.evaluation import zh_light as light
from qwen_hotword.training.zh_en_common import digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_zh_light_delivery as job  # noqa: E402


def test_export_checks_caches_and_keeps_full_causal_timelines(tmp_path: Path) -> None:
    candidates = [{"word": str(i), "phoneme": "a", "score": 1 - i / 10} for i in range(7)]
    result = {"steps": [[], candidates, []], "raw_last": ["ordinary"]}
    groups: dict[str, list[dict[str, Any]]] = {}
    totals: dict[str, dict[str, Counter[str]]] = {}
    for g in light.GROUPS:
        s = dict(
            group=g,
            key=g + "@@zh",
            sample_id="shared",
            language="zh",
            duration_seconds=4.1,
            audio_sha256="abc",
            audio_path="/data/example.wav",
            targets=[] if "n1" in g else ["0"],
        )
        groups[g] = [s]
        base.score(result, s, totals)
    cache = job.cached(tmp_path, s)
    cache.parent.mkdir(parents=True)
    cache.write_text(
        json.dumps(
            dict(result=result, run="run", audio="abc", language="zh", digest=digest(result))
        )
    )
    metrics = {g: {m: base.rates(c) for m, c in totals[g].items()} for g in groups}
    output = job.export(tmp_path, groups, "run", metrics)
    with zipfile.ZipFile(output) as z:
        assert len(z.namelist()) == 3
        data = json.loads(z.read("delivery/stream_contexts.json"))
        manifest = json.loads(z.read("delivery/audio_manifest.json"))
        for g in groups:
            key = g + "@@zh"
            assert "targets" not in manifest[key]["shared"]
            for m, k in job.MODES.items():
                assert data[m][key]["shared"] == [[], candidates[:k], []]
    assert job.export(tmp_path, groups, "run", metrics) == output
    before = output.read_bytes()
    with pytest.raises(ValueError, match="corrupted or incompatible"):
        job.export(tmp_path, groups, "other", metrics)
    assert output.read_bytes() == before
