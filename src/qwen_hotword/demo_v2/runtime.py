"""Session state for three live passes over identical PCM; v1 stays immutable."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from qwen_hotword.demo.runtime import HAN, TABLE_SHA, Demo, recovered
from qwen_hotword.phonemes.mfa_v03_assets import sha
from qwen_hotword.training.zh_en_common import digest, read, rows


class Studio(Demo):
    def init_studio(self) -> None:
        self.selected = -1
        self.mic_extras: dict[str, Any] = {}
        self.removed: set[str] = set()
        self.mic_removed: set[str] = set()
        self.mic_pcm = b""
        self.history: list[dict[str, Any]] = []
        self.mic_history: list[dict[str, Any]] = []

    def select(self, i: int) -> None:
        if not -1 <= i < len(self.clips):
            raise ValueError("未知样例")
        if self.selected == -1:
            self.mic_extras = dict(self.extras)
            self.mic_removed = set(self.removed)
            self.mic_history = self.history
        self.configure(self.clips[i]["word"] if i >= 0 else "", {} if i >= 0 else self.mic_extras)
        self.history = [] if i >= 0 else self.mic_history
        self.removed = set() if i >= 0 else set(self.mic_removed)
        self.selected = i

    def remove(self, word: str) -> dict[str, Any]:
        if word not in self.extras:
            raise ValueError("此词不在本场新增列表")
        self.configure(self.omit, {w: e for w, e in self.extras.items() if w != word})
        self.removed.add(word)
        return {"background": any(e.surface == word for e in self.entries)}

    def replay(self) -> bytes:
        pcm = self.pcm_for(self.clips[self.selected]) if self.selected >= 0 else self.mic_pcm
        if not pcm:
            raise ValueError("请先录制并定稿一句话")
        return pcm

    def begin_pass(self, kind: str) -> None:
        if kind not in ("replay", "mic") or (kind == "mic" and self.selected != -1):
            raise ValueError("音频来源与当前模式不符")
        self.expected = hashlib.sha256(self.replay()).hexdigest() if kind == "replay" else ""
        self.start()
        self.ever: set[str] = set()

    def record_step(self, data: bytes, final: bool) -> dict[str, Any]:
        r = self.step(data, final)
        self.ever.update(c["word"] for c in r["candidates"])
        if final:
            key = hashlib.sha256(self.pcm).hexdigest()
            if self.expected and key != self.expected:
                raise ValueError("重播音频与原声不一致")
            if not self.expected:
                self.removed.clear()  # A new microphone recording starts its own comparison.
                self.mic_pcm = bytes(self.pcm)
                self.history = []
            self.history.append(
                dict(
                    audio=key,
                    extras=sorted(self.extras),
                    removed=sorted(self.removed),
                    left=r["left"],
                    right=r["right"],
                    ever=sorted(self.ever),
                )
            )
            self.history = self.history[-100:]
        return r

    def snapshot(self) -> dict[str, Any]:
        return dict(
            clips=[
                dict(
                    id=c["sample_id"],
                    word=c["word"],
                    source=c["demo_source"],
                    seconds=c["after"]["seconds"],
                )
                for c in self.clips
            ],
            selected=self.selected,
            size=len({e.surface for e in self.entries}),
            extras={w: e.pronunciation for w, e in self.extras.items()},
            history=self.history,
            replayable=self.selected >= 0 or bool(self.mic_pcm),
        )

    def hunt(self, limit: int = 150) -> None:
        cache = self.root / "outputs/zh_en183_480_joint2351_eleven_v1"
        report = read(cache / "report.json")
        if report["status"] != "completed" or report["checkpoint_sha256"] != self.checkpoint:
            raise ValueError("历史缓存与演示模型不一致")
        policy = read(self.root / "configs/aishell_review_1054.v1.json")
        allowed = set(policy["candidate_surfaces"])
        excluded = {(x["utt_id"], x["word"]) for x in policy["excluded_target_pairs"]}
        source = read(self.root / "configs/zh_en_stream2351_probe.workzone.json")
        # Resume only when inputs, model identity, runtime code and decode settings match.
        train = self.root / "outputs/zh_en_183_noise70_dual_v2"
        paths = [
            train / "model_identity.json",
            train / "contract.json",
            self.root / source["previous_subset"],
            Path(source["home_meta"]),
            Path(__file__),
            Path(__file__).parents[1] / "demo/runtime.py",
            Path(__file__).parents[1] / "demo/decoder.py",
        ]
        import os
        import subprocess

        dep = subprocess.check_output(
            [
                os.environ["DEMO_DECODER_PY"],
                "-c",
                'from importlib.metadata import version; print(version("vllm"),version("torch"))',
            ],
            text=True,
        ).strip()
        identity = digest(
            [
                self.checkpoint,
                TABLE_SHA,
                policy,
                source,
                report["plan_sha256"],
                [sha(p) for p in paths],
                self.kw,
                dep,
                version("qwen-asr"),
                version("torch"),
            ]
        )
        folder = self.root / "outputs" / "zh_rag_v2_verified" / identity
        folder.mkdir(parents=True, exist_ok=True)
        saved = folder / "manifest.json"
        if saved.exists():
            rec = read(saved)
            if rec["identity"] != identity or digest(rec["body"]) != rec["digest"]:
                raise ValueError("样例缓存身份校验失败")
            clips = rec["body"]["clips"]
            for c in clips:
                self.pcm_for(c)
            attempted = rec["body"]["attempted"]
            for sid, w, path, audio in attempted:
                if sha(Path(path)) != audio:
                    raise ValueError(f"样例候选音频已变化：{sid}")
                self.checked.add((sid, w))
            self.clips = clips
            if self.clips:
                (self.out / "verified_cases.json").write_text(
                    json.dumps(clips, ensure_ascii=False, indent=2)
                )
            print(f"复用已核验样例 {len(clips)}/10；已检查 {len(attempted)} 条候选", flush=True)
            if len(clips) >= 10 or len(attempted) >= limit or rec["body"].get("exhausted"):
                return
        else:
            attempted = []

        def persist(exhausted: bool = False) -> None:
            body = dict(clips=self.clips, attempted=attempted, exhausted=exhausted)
            with NamedTemporaryFile(mode="w", dir=folder, delete=False, encoding="utf8") as tmp:
                json.dump(
                    dict(identity=identity, body=body, digest=digest(body)), tmp, ensure_ascii=False
                )
            Path(tmp.name).replace(saved)

        pool = []
        for r in rows(self.root / source["previous_subset"]):
            if not 2 < r["duration_seconds"] <= 15:
                continue
            for w in r["targets"]:
                if (
                    2 <= len(w) <= 12
                    and w in allowed
                    and HAN.fullmatch(w)
                    and (r["sample_id"], w) not in excluded
                    and w in "".join(r["reference_text"].split())
                ):
                    pool.append((w, dict(**r, demo_source="AISHELL 中文关键词子集")))
        home = []
        for row in rows(Path(source["home_meta"])):
            if (
                not re.fullmatch(r"val_v[12]_zh_\d+", row["id"])
                or row["status"] != "ok"
                or not 2 < float(row["dur_sec"]) <= 15
            ):
                continue
            for word in row["keywords"]:
                if len(word) >= 2 and HAN.fullmatch(word) and word in row["text"]:
                    home.append(
                        (
                            word,
                            dict(
                                sample_id=row["id"],
                                audio_path=row["wav"],
                                reference_text=row["text"],
                                demo_source="中文智慧家庭验证集",
                            ),
                        )
                    )

        def diverse(pairs: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
            counts: Counter[str] = Counter()
            output = []
            for word, row in sorted(pairs, key=lambda p: (-len(p[0]), p[1]["sample_id"])):
                output.append((counts[word], word, row))
                counts[word] += 1
            return [(w, r) for _, w, r in sorted(output, key=lambda x: x[0])]

        tried = len(attempted)
        ordered = (
            item
            for pair in zip_longest(diverse(home), diverse(pool))
            for item in pair
            if item is not None
        )
        for word, r in ordered:
            key = (r["sample_id"], word)
            if key in self.checked or any(
                c["word"] == word or c["sample_id"] == r["sample_id"] for c in self.clips
            ):
                continue
            if "audio_sha256" not in r:
                r["audio_sha256"] = sha(Path(r["audio_path"]))
            audio = r["audio_sha256"]
            path = cache / "audio_results/zh" / audio[:2] / (audio + ".json")
            rec = read(path)
            body = rec["result"]
            if (rec["audio"], rec["language"], rec["run"], rec["digest"]) != (
                audio,
                "zh",
                report["plan_sha256"],
                digest(body),
            ):
                raise ValueError(f"缓存校验失败：{path}")
            if not any(c["word"] == word for step in body["steps"] for c in step):
                continue
            self.checked.add(key)
            tried += 1
            print(f"验证示例 {tried}/{limit}: {key}", flush=True)
            pcm = self.pcm_for(r)
            self.configure(word, {})
            before, _ = self.run_clip(pcm)
            after = None
            words: list[str] = []
            if word not in before["left"] and word not in before["right"]:
                self.add(word)
                after, words = self.run_clip(pcm)
                if recovered(word, before, after, words):
                    self.clips.append(dict(**r, word=word, before=before, after=after))
                    (self.out / "verified_cases.json").write_text(
                        json.dumps(self.clips, ensure_ascii=False, indent=2)
                    )
            with (self.out / "case_checks.jsonl").open("a", encoding="utf-8") as f:
                print(
                    json.dumps(
                        dict(
                            id=key[0],
                            word=word,
                            source=r["demo_source"],
                            before=before,
                            after=after,
                            injected=word in words,
                        ),
                        ensure_ascii=False,
                    ),
                    file=f,
                )
            attempted.append([key[0], word, r["audio_path"], audio])
            persist()
            if len(self.clips) >= 10 or tried >= limit:
                break
        else:
            persist(exhausted=True)
