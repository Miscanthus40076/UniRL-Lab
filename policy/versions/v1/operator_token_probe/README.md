# Operator Token Probe Policy

Frozen snapshot note:
- This document belongs to the frozen experiment snapshot `policy/versions/v1/operator_token_probe/`.
- New iterations must be copied into a new version directory before modification.

## Purpose
`operator_token_probe` 是一个离线辅助策略，用 frozen Capacity V2 Dreamer 产出的关键 transition 自监督学习离散 operator token。

## Current Status
- 已支持 token posterior / prior / effect model / inverse action probe
- 已支持 capacity-mask 训练
- 不接 actor，不改 reward，不更新 Dreamer 主干

## Supported Observations
- 不直接消费原始 env 观测做决策
- 训练时消费由 frozen Dreamer world model 提取的 `augmented_feat`

## Supported Actions
- 训练时读取 replay / rollout 中的连续动作
- `act()` 仅返回零动作占位，不用于在线控制

## Config Example
```yaml
policy:
  type: operator_token_probe
  version: v1
  operator_token_probe:
    device: cuda
    allow_cpu_fallback: false
    source_checkpoint: exam/test_metaworld_dreamerv3_peg_insert_side_thick_event_capacity_v2_smoke/policy.ckpt
    output_dir: exam/test_metaworld_dreamerv3_peg_insert_side_operator_token_probe_capacity_v2_smoke/output/operator_token_probe/seed_0
    collect_steps: 2000
    train_steps: 3000
    batch_size: 32
    seq_len: 32
    learning_rate: 3.0e-4

operator_token:
  enabled: true
  num_tokens: 8
  token_dim: 32
  hidden_dim: 128
  gumbel_temperature: 1.0
  straight_through: true
  train_on_capacity_only: true
  effect_loss_scale: 1.0
  inverse_loss_scale: 0.5
  prior_loss_scale: 0.1
  usage_entropy_scale: 0.01
  detach_features: true
```

## Known Limitations
- 第一版不使用 replay 持久化，数据由 frozen Dreamer 重新 rollout 收集
- token 只做辅助诊断，不接策略先验或记忆模块
- 图像样例默认只保存 JSONL 索引，不保证每个 token 都生成 GIF
