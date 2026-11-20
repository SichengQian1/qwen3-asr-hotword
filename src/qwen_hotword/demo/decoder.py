"""Private socket to one native vLLM engine, two independent Qwen streaming states."""

from __future__ import annotations

import sys
from importlib.metadata import version
from multiprocessing.connection import Connection


def serve(fd: int, model: str) -> None:

    with Connection(fd) as channel:
        try:
            import numpy as np
            from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

            if version("vllm").split("+")[0] != "0.23.0" or version("qwen-asr") != "0.0.6":
                raise RuntimeError("需要已验证的 vLLM 0.23.0 / qwen-asr 0.0.6 环境")
            engine = LLM(
                model=model,
                dtype="float16",
                enforce_eager=True,
                max_num_seqs=1,
                max_model_len=4096,
                gpu_memory_utilization=0.25,
                seed=42,
                generation_config="vllm",
                enable_prefix_caching=False,
            )
            if engine.apply_model(lambda m: type(m).__module__) != [
                "vllm.model_executor.models.qwen3_asr"
            ]:
                raise RuntimeError("未加载 vLLM 原生 Qwen3-ASR 实现")
            sampling = SamplingParams(temperature=0, max_tokens=128, seed=42)
            channel.send({"ok": True})
            while True:
                req = channel.recv()
                audio = np.frombuffer(req["wave"], dtype="<f4")
                generated = engine.generate(
                    {"prompt_token_ids": req["ids"], "multi_modal_data": {"audio": (audio, 16000)}},
                    sampling_params=sampling,
                    use_tqdm=False,
                )
                channel.send({"text": generated[0].outputs[0].text})
        except EOFError:
            pass
        except Exception as exc:
            channel.send({"error": str(exc)})
            raise


if __name__ == "__main__":
    serve(int(sys.argv[1]), sys.argv[2])
