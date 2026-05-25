# DreamerV3 Architecture

说明：
- 本文件描述 `policy/dreamerv3/` 工作区实现。
- 已用于实验的冻结实现位于 `policy/versions/<version>/dreamerv3/`。
- 新迭代必须复制冻结版本后继续修改，不能原地改历史实验版本。

整体流程：

`obs -> policy wrapper preprocess -> encoder -> RSSM -> actor/value/world-model heads`

模块分层：

- `config.py`
  负责 Dreamer wrapper 参数 dataclass、校验与构建
- `processor.py`
  负责 obs 预处理与 action 后处理
- `dreamerv3_policy_impl.py`
  负责工程适配、latent reset、单步 transition 写入 replay、checkpoint 装载保存
- `dreamerv3_agent.py`
  负责 world model / actor / value 的训练与 imagined rollout，以及在线更新节奏（warmup/train_every/train_ratio）
- `dreamerv3_model.py`
  负责配置 dataclass 与 obs spec 建模

Dependency direction：

`policy_impl -> {config, processor, agent} -> model`

Lower layers must not depend on upper layers.

输入路径：

- vector:
  `[obs_dim] -> MLPEncoder(embed_dim)`
- image:
  `[C,H,W] -> CNNEncoder(embed_dim)`
- multi-camera:
  `[(H,W,C), ...] -> channel concat -> [C_total,H,W]`

Tensor shape convention：

- image obs:
  `[B,T,C,H,W]`
- vector obs:
  `[B,T,obs_dim]`
- encoder output:
  `[B,T,embed_dim]`
- RSSM deter:
  `[B,T,deter_dim]`
- RSSM stoch:
  `[B,T,stoch_dim]`
- actor input:
  `[B,T,deter+stoch]`
- action:
  `[B,T,action_dim]`

核心维度默认值：

- embed: `128`
- RSSM deter: `128`
- RSSM stoch: `32`
- RSSM classes: `32`
- shared hidden: `256`
- actor hidden: `256 x 2`
- value hidden: `256 x 2`

Replay / update：

- runner 每步调用 `policy.update(step_batch)`
- policy 把单步数据写进 `EpisodeReplayBuffer`
- policy 调用 `agent.update_from_replay(...)`
- agent 负责在线训练节奏控制与序列采样：
  - `obs[t]` 为当前观测
  - `action[t]` 为前一时刻动作，起点补零
  - `is_first[0]=1` 仅在序列从 episode 起点开始时置位
- 先训练 world model，再做 imagined actor/value 更新

World model losses：

- `reconstruction_loss`
- `reward_loss`
- `continuation_loss`
- `dynamics_kl_loss`
- `representation_kl_loss`

DreamerV3-style targets and transforms：

- vector observations use `symlog` before the MLP encoder
- vector reconstructions decode back from symlog space
- reward prediction defaults to a symlog two-hot head
- value prediction defaults to a symlog two-hot head
- RSSM stochastic state is categorical with shape `[B,T,stoch_dim,stoch_classes]`
- RSSM feature flattens stochastic categories before concatenating with deterministic state
- RSSM sampling supports `unimix` smoothing
- KL training supports `free_nats`
- KL training supports weighted dyn/rep balancing via `kl_balance`
- value training supports a slow target network
- replay training supports `repval_loss` on posterior features
- actor training supports running `advantage_norm`
- value targets support running `return_norm`
- replay sampling supports optional priority weighting
- policy update loop supports internal multi-update budgeting via `train_ratio`

Replay sequence convention：

Stored fields：

- `obs`
- `action`
- `reward`
- `done`
- `is_first`

Sequence layout：

- `obs[t]` -> current observation
- `action[t]` -> previous action
- `reward[t]` -> reward after `action[t]`
- `done[t]` -> termination after `action[t]`

Latent reset convention：

- latent state must reset on `env.reset()`
- latent state must reset when `done=True`
- `is_first=1` forces RSSM reset during sequence training

版本记录：

- `v0`
  从 `policy/REF_policy/dreamerv3` 迁移到 `policy/dreamerv3`
- `v1`
  拆成 wrapper / agent / model 三层，接入当前 factory 和 runner
- `v2`
  引入更多 DreamerV3 风格训练细节：split KL、free nats、symlog、two-hot reward/value
- `v3`
  引入离散 RSSM、unimix、slow value、repval loss
- `v4`
  引入 return normalization 与 advantage normalization
- `v5`
  引入 priority replay 与 train-ratio 驱动的多次内部更新
- `v6`
  按 BYOP 约定重构：新增 `config.py` 和 `processor.py`，将裸字典配置读取与 obs/action 变换从 wrapper 拆分
- `v7`
  引入版本冻结约定：工作区实现与 `policy/versions/<version>/dreamerv3/` 实验快照分离
