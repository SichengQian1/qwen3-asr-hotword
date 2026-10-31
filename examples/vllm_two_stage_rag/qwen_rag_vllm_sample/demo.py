"""Dependency-free CONTROL-FLOW demo. All audio/model/CTC/text values are simulated."""

from __future__ import annotations

import json
from typing import Any

from qwen_rag_vllm_sample.contracts import AudioCache, build_prompt, owner_tag


class SimulatedWorker:
    def __init__(self) -> None:
        self.cache = AudioCache()
        self.encoder_calls = 0
        self.generation_calls = 0
        self.events: list[str] = []

    def prepare(self, request_id: str) -> dict[str, int]:
        self.events.append("encode_and_ctc_before_any_generation_request")
        self.encoder_calls += 1
        owner = owner_tag(request_id)
        # Stand-in for NPU-resident Z[13,2048], not actual model features.
        key = self.cache.put(
            "simulated_device_embedding",
            owner=owner,
            feature_length=100,
            byte_size=13 * 2048 * 2,
        )
        return {"handle": key, "owner": owner, "feature_length": 100}

    def generate(self, prepared: dict[str, int], prompt: str) -> str:
        self.events.append("create_final_request_then_read_cached_audio_then_decode")
        self.cache.get(
            prepared["handle"],
            owner=prepared["owner"],
            feature_length=prepared["feature_length"],
        )
        self.generation_calls += 1
        if "Copacabana" not in prompt:
            raise ValueError("demo expected the selected term in the FINAL prompt")
        return "SIMULATED ONLY: Vamos para Copacabana."


def demonstrate(*, fail_generation: bool = False) -> tuple[dict[str, Any], SimulatedWorker]:
    worker = SimulatedWorker()
    prepared = worker.prepare("demo-request-1")
    try:
        worker.events.append("retrieve_topk_on_cpu")
        selected = ["Copacabana"]  # Simulated CTC retrieval, NOT a real recognition.
        prompt = build_prompt("Spelling references, only if spoken: " + ", ".join(selected), "pt")
        worker.events.append("final_prompt_ready")
        if fail_generation:
            raise RuntimeError("simulated decoder error")
        text = worker.generate(prepared, prompt)
    finally:
        worker.cache.release(prepared["handle"], owner=prepared["owner"])
        worker.events.append("release_sample_owned_audio_cache")
    report = {
        "scope": "SIMULATED_CONTROL_FLOW_ONLY_NO_MODEL_NO_NPU_NO_QUALITY_OR_MEMORY_MEASUREMENT",
        "encoder_calls": worker.encoder_calls,
        "generation_requests": worker.generation_calls,
        "audio_cache_entries_after_release": len(worker.cache.entries),
        "selected_hotwords_simulated": selected,
        "final_prompt": prompt,
        "transcript_simulated": text,
        "events": worker.events,
    }
    return report, worker


def main() -> None:
    report, _worker = demonstrate()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
