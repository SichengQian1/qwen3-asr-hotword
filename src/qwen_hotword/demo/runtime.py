from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from importlib.metadata import version
from itertools import zip_longest
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from qwen_hotword.evaluation import aishell_run as ar
from qwen_hotword.evaluation import zh_en_delivery as joint
from qwen_hotword.evaluation.aishell_reviewed import assemble_entries
from qwen_hotword.hotwords.anchor_index import PhonemeAnchorIndex
from qwen_hotword.hotwords.registry import HotwordEntry
from qwen_hotword.inference import external_keyword_retrieval as external
from qwen_hotword.inference.external_keyword_topk_replay import _select
from qwen_hotword.inference.hotword_prompt import build_hotword_prompt
from qwen_hotword.inference.reviewed_joint import collapse_variants
from qwen_hotword.inference.stream_context_contract import candidates
from qwen_hotword.inference.streaming_core import refresh_streaming_prompt
from qwen_hotword.phonemes.mfa_v03_assets import pin, run_worker, sha
from qwen_hotword.training.zh_en_common import completed, digest, read, rows, save_rows
from qwen_hotword.training.zh_en_labels import frontend, install_frontend, make_worker

CHUNK = 64000  # bytes, 2s mono S16LE at 16 kHz
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{1,24}")
TABLE_SHA = "33e73fa9df358475d766b7cd3c5f2fa07071a5ffc25ed8e87923e2effaae5e40"


def recovered(
    word: str, before: dict[str, Any], after: dict[str, Any], timeline: list[str]
) -> bool:
    return (
        word not in before["left"]
        and word not in before["right"]
        and before["left"] == after["left"]
        and word in after["right"]
        and word in timeline
    )


class Demo:
    def __init__(self, root: Path, output: Path, gpu: int, decoder_python: Path) -> None:
        self.root, self.out = root, output
        self.clips: list[dict[str, Any]] = []
        self.checked: set[tuple[str, str]] = set()
        self.extras: dict[str, HotwordEntry] = {}
        self.omit = ""
        self.seq = 0
        cfg = read(root / "configs/aishell_original_retrieval.workzone.json")
        ar.check_gpu(gpu, 55)
        model, train, foundation, self.vocab = ar.identity(
            root, cfg, read(root / cfg["review_policy"])
        )
        self.checkpoint = model["checkpoint_sha256"]
        path = root / "configs/zh_en_joint2351.v1.jsonl"
        pin(path, TABLE_SHA)
        self.base, self.base_owners = joint.registry(path, self.vocab)
        runtime = root / cfg["training_run"] / "frontend_runtime"
        if completed(runtime, train["frontend_packages"]) is not None:
            sys.path.insert(0, str(runtime))
        else:
            install_frontend(output, train["frontend_packages"])
        self.worker = make_worker(root, output, foundation, train)
        self.det = external._load_detector(
            model=Path(model["model_path"]),
            checkpoint=Path(model["checkpoint_path"]),
            vocab=self.vocab,
            hotwords=self.base,
            device="cuda:0",
            dtype="bfloat16",
            language="English",
        )
        self.det.language = "Chinese"
        self.entries, self.owners = self.base, self.base_owners
        a, b = socket.socketpair()
        self.conn = Connection(a.detach())
        try:
            self.child = subprocess.Popen(
                [
                    str(decoder_python),
                    "-u",
                    "-m",
                    "qwen_hotword.demo.decoder",
                    str(b.fileno()),
                    model["model_path"],
                ],
                pass_fds=(b.fileno(),),
                env=dict(os.environ),
            )
        finally:
            b.close()
        try:
            self.reply(600)
            from qwen_asr import Qwen3ASRModel

            if version("qwen-asr") != "0.0.6":
                raise ValueError("流式状态机需要 qwen-asr 0.0.6")
            self.backend = Qwen3ASRModel(
                backend="vllm", model=self, processor=self.det.wrapper.processor
            )
            self.kw: dict[str, Any] = dict(
                language="Chinese", unfixed_token_num=5, unfixed_chunk_num=2, chunk_size_sec=2.0
            )
        except BaseException:
            self.close()
            raise

    def generate(self, inputs: list[dict[str, Any]], **kwargs: Any) -> list[Any]:
        # Only the official SDK's single-audio request crosses this private process boundary.
        (req,) = inputs
        (wave,) = req["multi_modal_data"]["audio"]
        ids = self.backend.processor.tokenizer.encode(req["prompt"], add_special_tokens=False)
        self.conn.send({"ids": ids, "wave": wave.astype("<f4", copy=False).tobytes()})
        return [SimpleNamespace(outputs=[SimpleNamespace(text=self.reply()["text"])])]

    def close(self) -> None:
        self.conn.close()
        try:
            self.child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.child.terminate()
            self.child.wait(timeout=30)

    def reply(self, timeout: int = 120) -> dict[str, Any]:
        if not self.conn.poll(timeout):
            self.conn.close()  # do not consume a stale response on the next request
            raise TimeoutError("解码超时；请重新启动服务")
        result: dict[str, Any] = self.conn.recv()
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def configure(self, omit: str, extras: dict[str, HotwordEntry]) -> None:
        entries = tuple(
            e for e in self.base if e.surface != omit and e.surface not in extras
        ) + tuple(extras.values())
        owners = {e.hotword_id: self.base_owners.get(e.hotword_id, e) for e in entries}
        index = PhonemeAnchorIndex(entries, config=self.det.anchor_index.config)
        self.entries, self.owners, self.omit, self.extras = entries, owners, omit, dict(extras)
        self.det.hotword_by_id = {e.hotword_id: e for e in entries}
        self.det.anchor_index = index

    def add(self, word: str) -> dict[str, Any]:
        if not HAN.fullmatch(word):
            raise ValueError("请输入 1–24 个汉字。")
        if word not in self.extras and len(self.extras) >= 32:
            raise ValueError("单场最多 32 个临时词，请重置后再加。")
        units = frontend(word, "zh")
        folder = self.out / "words" / digest(word)
        folder.mkdir(parents=True, exist_ok=True)
        req = folder / "word_request.jsonl"
        dest = folder / "word_prediction.jsonl"
        save_rows(
            req,
            [
                {"language": lang, "word": w}
                for lang, w in sorted({(u["language"], u["word"]) for u in units})
            ],
        )
        run_worker(self.root, *self.worker, req, dest)
        found, errors = assemble_entries(
            [dict(surface=word, units=units, errors=[])], list(rows(dest)), self.vocab
        )
        if not found or errors:
            raise ValueError(f"MFA 未生成完整发音：{errors}")
        entry = replace(found[0], hotword_id="demo:" + word)
        self.configure(self.omit, {**self.extras, word: entry})
        return {"word": word, "phones": entry.pronunciation}

    def start(self) -> None:
        self.pcm = bytearray()
        self.last_candidates: list[dict[str, Any]] = []
        self.seq = 0
        self.states = [self.backend.init_streaming_state(context="", **self.kw) for _ in range(2)]

    def step(self, pcm: bytes, final: bool) -> dict[str, Any]:
        import numpy as np

        if (
            not final
            and len(pcm) != CHUNK
            or len(pcm) > CHUNK
            or len(pcm) % 2
            or len(self.pcm) + len(pcm) > CHUNK * 15
            or not self.pcm
            and not pcm
        ):
            raise ValueError("无效的音频块／超过 30 秒")
        self.pcm.extend(pcm)
        if pcm:
            wave = np.frombuffer(bytes(self.pcm), dtype="<i2").astype("float32") / 32768
            raw = external._retrieve_waveform(self.det, wave, self.vocab, saved_raw_rank_depth=64)
            ranked = collapse_variants(raw, self.entries, self.owners, joint.GATE)[
                "raw_ranked_matches"
            ]
            self.last_candidates = candidates(_select(ranked, gate=joint.GATE, top_k=7))
        prompt = build_hotword_prompt(
            [c["word"] for c in self.last_candidates],
            template="以下是可能出现的词语，仅供转写参考：{hotwords}",
        )
        refresh_streaming_prompt(self.backend, self.states[1], prompt=prompt, **self.kw)
        chunk = np.frombuffer(pcm, dtype="<i2").astype("float32") / 32768
        for state in self.states:
            if len(chunk):
                self.backend.streaming_transcribe(chunk.copy(), state)
            if final:
                self.backend.finish_streaming_transcribe(state)
        result = {"left": self.states[0].text, "right": self.states[1].text}
        self.seq += 1
        return dict(
            **result, candidates=self.last_candidates, seconds=len(self.pcm) / 32000, final=final
        )

    def pcm_for(self, row: dict[str, Any]) -> bytes:
        import soundfile as sf

        path = Path(row["audio_path"])
        pin(path, row["audio_sha256"])
        wave, sr = sf.read(str(path), dtype="int16")
        if sr != 16000 or wave.ndim != 1 or not (0 < len(wave) <= 480000):
            raise ValueError("样例需要 16 kHz 单声道、至多 30 秒")
        return bytes(wave.astype("<i2").tobytes())

    def run_clip(self, pcm: bytes) -> tuple[dict[str, Any], list[str]]:
        self.start()
        words: list[str] = []
        result = {}
        for offset in range(0, len(pcm), CHUNK):
            result = self.step(pcm[offset : offset + CHUNK], offset + CHUNK >= len(pcm))
            words.extend(c["word"] for c in result["candidates"])
        return result, words

    def hunt(self, limit: int = 24) -> None:
        cache = self.root / "outputs/zh_en183_480_joint2351_eleven_v1"
        report = read(cache / "report.json")
        if report["status"] != "completed" or report["checkpoint_sha256"] != self.checkpoint:
            raise ValueError("历史缓存与演示模型不一致")
        policy = read(self.root / "configs/aishell_review_1054.v1.json")
        allowed = set(policy["candidate_surfaces"])
        excluded = {(x["utt_id"], x["word"]) for x in policy["excluded_target_pairs"]}
        source = read(self.root / "configs/zh_en_stream2351_probe.workzone.json")
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

        tried = 0
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
            if len(self.clips) >= 2 or tried >= limit:
                break
