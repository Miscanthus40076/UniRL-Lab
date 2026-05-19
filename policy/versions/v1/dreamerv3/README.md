# DreamerV3 Policy

Frozen snapshot note:
- This document belongs to the frozen experiment snapshot `policy/versions/v1/dreamerv3/`.
- New iterations must be copied to a new version directory instead of editing this snapshot in place.

用于把参考版 DreamerV3 接入当前工程的 `policy/` 统一接口。

当前状态：
- 已通过 `policy/registry.py` 注册到策略工厂
- 可被 `scripts/train.py` / `scripts/run_exam.py` 通过 `policy.type + policy.version` 创建
- 训练细节封装在策略内部 replay + 序列采样中
- 已按统一策略约定拆分为 `config.py` / `processor.py` / `dreamerv3_policy_impl.py` / `dreamerv3_agent.py` / `dreamerv3_model.py`
- 已补入一批 DreamerV3 风格训练细节：discrete RSSM、split KL、free nats、symlog、two-hot reward/value、slow value、repval、return/advantage norm、priority replay、train ratio

支持的观测：
- 一维向量观测，自动 flatten，走 MLP encoder
- 单张图像观测，支持 HWC/CHW，走 CNN encoder
- 多相机 list/tuple 输入，只有元素确实是图像 frame 时才按通道拼接成单个 `[C,H,W]`

支持的动作：
- 连续动作
- 输出范围为 `tanh` 后的 `[-1, 1]`

已接入的训练方式：
- world model reconstruction / reward / continue / dynamics KL / representation KL
- imagined rollout actor-value 更新
- 离散 stochastic latent：`stoch_dim x stoch_classes`
- RSSM categorical posterior/prior with `rssm_unimix`
- 向量观测默认走 `symlog`
- reward / value 默认走 symlog two-hot 头
- value 默认支持 slow target network
- replay posterior features 默认支持 `repval_loss`
- actor/value 更新支持 `return_norm` 与 `advantage_norm`
- replay 可选 priority sampling
- 每个 env step 可按 `train_ratio` 触发多次内部更新
- 可选 `contact_mode` / `is_grasping` 辅助头，默认关闭

已知限制：
- 当前 runner 仍以单环境、单步 transition 驱动，Dreamer 的序列训练在策略内部自行缓冲
- replay 现在显式存储 `obs_t` 对应的 `prev_action_t`，避免在采样时右移动作导致 latent / reward / done 时间步错位
- checkpoint 目前保存网络、优化器和 wrapper 元数据，但不保存 replay 内容
- 多相机模式要求所有相机分辨率一致
- 当前 `dm_control` 默认环境仍返回向量观测；图像路径只有在环境层实际提供图像时才算完成端到端验证
- 目前仍是简化版 PyTorch Dreamer，而不是官方 JAX runtime 的完整等价迁移
- 还未补齐官方实现中的更完整 value normalization 变体、AGC optimizer、以及更接近官方的分布式/异步训练调度
