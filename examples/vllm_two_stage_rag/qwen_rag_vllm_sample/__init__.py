"""Example only: import does not load Torch, vLLM, model weights or a device."""

ARCHITECTURE = "Qwen3ASRTwoStageForConditionalGeneration"


def register() -> None:
    """vLLM general plugin: runs in the processes that need model registration."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        ARCHITECTURE,
        "qwen_rag_vllm_sample.model:Qwen3ASRTwoStageForConditionalGeneration",
    )
