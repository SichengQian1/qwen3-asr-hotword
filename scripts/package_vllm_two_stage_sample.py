"""Package the explicit colleague sample and minimal project helpers; no model/device use."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="new directory; never overwritten"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    package_root = "qwen_vllm_two_stage_sample"
    files = {
        path.relative_to(root).as_posix(): path
        for path in (root / "examples/vllm_two_stage_rag").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    for relative in (
        "docs/VLLM_TWO_STAGE_CTC_RAG_SAMPLE.md",
        "tests/test_vllm_two_stage_sample.py",
        "configs/phonemes/en_es_ptbr_precision_ipa_vocab.v0.3.json",
    ):
        files[relative] = root / relative
    helpers = (
        "__init__.py",
        "hotwords/__init__.py",
        "modeling/__init__.py",
        "phonemes/__init__.py",
        "training/__init__.py",
        "hotwords/anchor_index.py",
        "hotwords/exact_automaton.py",
        "hotwords/registry.py",
        "hotwords/scoring.py",
        "phonemes/coverage.py",
        "modeling/ctc_head.py",
        "training/edit_distance.py",
    )
    for relative in helpers:
        files[f"project_subset/src/qwen_hotword/{relative}"] = root / "src/qwen_hotword" / relative
    contents = {relative: path.read_bytes() for relative, path in sorted(files.items())}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    identity = {
        "scope": "colleague_source_sample_no_models_no_hardware_result",
        "repository": "https://github.com/SichengQian1/qwen3-asr-hotword",
        "base_git_commit": commit,
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
    }
    contents["SOURCE_IDENTITY.json"] = (json.dumps(identity, indent=2) + "\n").encode()
    contents["START_HERE.md"] = (
        "# 两阶段 vLLM + CTC 热词 RAG 接入样例\n\n"
        "先读 [中文完整说明](docs/VLLM_TWO_STAGE_CTC_RAG_SAMPLE.md)。\n\n"
        "代码位于 examples/vllm_two_stage_rag；最小项目帮助模块位于 project_subset/src。\n"
        "真实 vLLM/310P 路径尚未实机验证。模拟输出不能当作质量、性能或显存结果。\n\n"
        "在本解压目录可运行：\n\n```bash\n"
        "PYTHONPATH=project_subset/src:examples/vllm_two_stage_rag "
        "python -B -m qwen_rag_vllm_sample.demo\n```\n"
    ).encode()
    bundle = args.output / "qwen_vllm_two_stage_sample.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, (relative, data) in enumerate(sorted(contents.items()), 1):
            archive.writestr(f"{package_root}/{relative}", data)
            print(f"Package colleague sample: {index}/{len(contents)}", flush=True)
    checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
    (args.output / "sha256.txt").write_text(f"{checksum}  {bundle.name}\n", encoding="utf-8")
    print(json.dumps({"status": "packaged", "zip": str(bundle), "sha256": checksum}, indent=2))


if __name__ == "__main__":
    main()
