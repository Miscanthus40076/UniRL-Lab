# Operator Token Probe Architecture

## Overall Pipeline
1. 加载 frozen Capacity V2 Dreamer checkpoint
2. 在源环境上 rollout，收集 sequence replay
3. 用 frozen world model 计算 `augmented_feat_seq` 和 `event_capacity_gate`
4. 在 capacity transition 上训练 operator token posterior / prior / effect / inverse probe
5. 输出 summary、usage、examples 和 probe checkpoint

## Modules
- `processor.py`
  负责把 transition batch 转成设备张量
- `operator_token_probe_model.py`
  只放 token posterior / prior / effect / inverse 的组合网络
- `operator_token_probe_agent.py`
  负责 loss、optimizer、metrics、save/load
- `operator_token_probe_policy_impl.py`
  负责 policy 边界和离线训练入口

## Tensor Shapes
- `x_t`: `[B, T-1, D]`
- `x_tp1`: `[B, T-1, D]`
- `delta_x`: `[B, T-1, D]`
- `action_t`: `[B, T-1, A]`
- `capacity_gate`: `[B, T-1, 1]`
- `token_logits`: `[B, T-1, K]`
- `token_onehot_st`: `[B, T-1, K]`
- `pred_delta_x`: `[B, T-1, D]`
- `pred_action`: `[B, T-1, A]`

## Version History
- V0
  - forward effect prediction
  - inverse action probe
  - prior prediction from `x_t, action_t`
  - capacity-only masked training
