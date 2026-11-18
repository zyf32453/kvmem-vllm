# Text-level Deletion 与 KV-level Deletion

从同一个 Agent 历史快照出发，研究删掉一段文字再重新 Prefill，以及让模型跳过该段历史 KV，分别如何影响后续答案、系统开销和可复用显存。

- [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) 记录现有 Graph Cleanup、vLLM 缓存路径、DeepSWE 评分与测试证据。
- [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) 定义五个实验组、两种对齐规则、指标、统计和阶段入口。

Phase 1 在 CPU 上准备可审计的重放输入与两种 Deletion Plan。Graph Cleanup 提供候选和源坐标，原 Trace 包用于校验，实际 tokenizer 提供原始 token IDs 和 offset mapping。后续推理需要在已验收的实验环境中绑定真实页表和模型状态。
