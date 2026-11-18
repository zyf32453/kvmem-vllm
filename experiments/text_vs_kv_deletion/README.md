# Text-level Deletion 与 KV-level Deletion

从同一个 Agent 历史快照出发，研究删掉一段文字再重新 Prefill，以及让模型跳过该段历史 KV，分别如何影响后续答案、系统开销和可复用显存。

- [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) 记录现有 Graph Cleanup、vLLM 缓存路径、DeepSWE 评分与测试证据。
- [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) 定义五个实验组、两种对齐规则、指标、统计和阶段入口。

Phase 1 在 CPU 上准备可审计的重放输入与两种 Deletion Plan。Graph Cleanup 提供候选和源坐标，原 Trace 包用于校验，实际 tokenizer 提供原始 token IDs 和 offset mapping。后续推理需要在已验收的实验环境中绑定真实页表和模型状态。

## 验证和生成数据

在仓库根目录使用 Python 3.12+ 运行。测试用合成数据和字符 tokenizer，可以在 CPU 上运行。使用真实数据时，用实验环境实际安装的 Transformers、Tokenizers 和对应模型的本地 tokenizer。

```sh
python -m unittest discover -s experiments/text_vs_kv_deletion -p 'test_*.py' -v
python -m unittest discover -s experiments/graph_cleanup -p 'test_*.py' -v
```

准备 Graph Cleanup 已生成的 `candidates.jsonl`、`source_hashes.json` 和 `snapshots/`，以及对应的原始 Agent Trace bundle。用下面的 JSON 结构记录计划使用的环境。示例参数对应 Graph Cleanup 的单序列先导，模型和 tokenizer revision 应替换为实际的 40 位提交 SHA 或 `sha256:` 加实际文件清单的 64 位 SHA-256。在 GPU 阶段核对实际运行配置。

```json
{
  "model": {"id": "Qwen3.8-27B-GPTQ-Int4", "revision": "<actual pinned revision>"},
  "tokenizer": {"id": "<actual tokenizer ID>", "revision": "<actual pinned revision>"},
  "vllm_version": "0.32.0",
  "kernel_page_tokens": 32,
  "attention_layers": 16,
  "recurrent_layers": 48,
  "recurrent_layout": "qwen_gdn",
  "inference_params": {
    "seed": 7231, "temperature": 0, "max_tokens": 768,
    "enable_prefix_caching": false, "enforce_eager": true,
    "max_num_seqs": 1, "max_num_batched_tokens": 512, "max_model_len": 34816,
    "dtype": "bfloat16", "quantization": "gptq_marlin", "kv_cache_dtype": "fp8_e4m3",
    "attention_config": {"backend": "FLASHINFER"}, "language_model_only": true
  }
}
```

```sh
python -m experiments.text_vs_kv_deletion.prepare build \
  --bundle artifacts/agent-traces-handoff-20261102-r2 \
  --dataset artifacts/graph-cleanup-4090-20261105-r4 \
  --tokenizer /path/to/qualified/tokenizer \
  --config /path/to/replay-config.json \
  --output artifacts/text-vs-kv-prepared

python -m experiments.text_vs_kv_deletion.prepare verify \
  --output artifacts/text-vs-kv-prepared \
  --bundle artifacts/agent-traces-handoff-20261102-r2
```

每次 build 使用新目录。输出 `manifest.json` 索引候选与两份计划并保存文件哈希。`snapshots/` 存原 Trace 来源消息、重放消息、Token IDs、字符和逻辑页映射。`plans/` 存每种模式的文本编辑坐标、重分词结果、逻辑页、附带删除量与无法对齐的原因。摘要里的 eligible 表示数据准备通过，实际推理还有独立 gate。

从 `human-review.csv` 复制一份到输出目录外再填写，使用 `build --reviews /path/to/reviews.csv` 写进下一份准备结果。标注包含 `label`、`reviewer` 和 `evidence`。未审核样本对应空标签，模型推理的正式样本需先完成人工审核。

## 运行时证据

离线快照中的物理映射和 Attention KV 哈希为 `null`，recurrent 状态标为待采集。后续 worker 用 `verify --capture /path/to/capture.json` 关联其运行时记录。捕获记录须包含以下字段。

| 层次 | 字段 |
| --- | --- |
| 快照与请求 | `snapshot_sha256`、`config_sha256`、`tokenizer_backend_sha256`、`cache_instance`、`request_generation`、`history_tokens` 和 `kernel_page_tokens` |
| Attention 层 | `attention` 以层名为键，每层提供原生 `slot_mapping`、`cache_page_count` 和 `kv_sha256` |
| 循环状态 | `recurrent_state` 包含 `layout`、`boundary_tokens`、逐层 `conv_sha256` 和 `ssm_sha256`、以及 `accepted_counter_sha256` |

校验器根据实际 slot mapping 计算每层逻辑到物理内核页的关系，并检查边界、重复页、越界槽位、历史和循环状态指纹的覆盖。结果仍需在同一 GPU 会话中由 worker 采集并校验状态字节，再用于正式测量。GPU runner、DeepSWE 任务和物理回收各有阶段验收条件。
