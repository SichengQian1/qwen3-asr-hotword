# 同一 vLLM 模型，两阶段音素 RAG 样例

这是一份交给集成同事的**接入原型**：音频先编码并运行 CTC，检索完成后才创建最终
Decoder 请求。音频表示保留在原 worker，第二阶段通过缓存编号引用；没有再加载一套
Qwen，也没有修改已运行请求的 Prompt。

完整说明见 [VLLM_TWO_STAGE_CTC_RAG_SAMPLE.md](../../docs/VLLM_TWO_STAGE_CTC_RAG_SAMPLE.md)。

| 文件 | 作用 |
|---|---|
| `qwen_rag_vllm_sample/demo.py` | 无依赖、无模型、无 NPU 的流程演示；内容全部为模拟值 |
| `qwen_rag_vllm_sample/contracts.py` | 请求绑定、音频长度、缓存生命周期、Prompt 格式 |
| `qwen_rag_vllm_sample/model.py` | vLLM 自定义模型、Processor、worker RPC、音频缓存旁路 |
| `qwen_rag_vllm_sample/retrieval.py` | 复用项目 Anchor、局部重排、词面合并和门限 |
| `qwen_rag_vllm_sample/run.py` | 真实设备单条音频三路径对照及逐 token 检查 |
| `config.example.json` | 最新 100 类配套资产及 SHA；路径需按同事环境修改 |
| `pyproject.toml` | vLLM 通用插件注册入口；不安装或升级 vLLM/Torch |

在仓库根目录可立即运行模拟：

```bash
PYTHONPATH=src:examples/vllm_two_stage_rag python -B -m qwen_rag_vllm_sample.demo
```

**验证边界：**本地通过流程/缓存/真实检索单测及静态检查；真实接入按 vLLM 和
vLLM Ascend **0.23.0** 源码编写，未在 310P/H200 加载或运行。因此本包不是“已经
在 310P 上跑通”的产品，也没有提供节省 10GB 或加速多少的实测结论。

压缩包另含 `project_subset/src`，是为该样例摘取的项目源文件及 100 类词表，方便同事
脱离完整仓库阅读/运行；完整仓库使用 `src`，解压包使用 `project_subset/src`。
两者选其一，不能把摘取的旧副本混入后续生产项目。具体依赖、安装与设备运行步骤
见完整说明。未包含模型、Head 权重、业务热词索引或业务音频。
