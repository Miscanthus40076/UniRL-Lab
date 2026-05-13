# Exam Config Convention

## YAML Template

```yaml
train:
  seed: 0
  total_steps: 10000
  max_episode_steps: null
  multi_task: false

  log_interval: 100
  save_interval: 1000

  eval_interval: 5000
  eval_episodes: 5

  record_interval: 100

  device: cpu

env:
  type: dmcontrol
  name: cartpole_swingup

  domain_name: cartpole
  task_name: swingup
  tasks: []

  observation:
    type: image
    num_cams: 1

  action:
    clip: true
    normalize: true

  render:
    enabled: false
    save_frames: false
    save_video: false
    backend: null
    backend_priority: [egl, osmesa]
    allow_software_render_fallback: false

  isaac:
    task_name: null
    num_envs: 1
    headless: true
    create_cam: null

policy:
  type: random
  name: random_baseline

  checkpoint:
    load: false
    path: null
    save: false

  random:
    action_low: -1.0
    action_high: 1.0
```

## Train

| Field | Description |
| --- | --- |
| `seed` | 随机种子 |
| `total_steps` | 总训练步数 |
| `max_episode_steps` | 单 `episode` 最大步数，可选 |
| `multi_task` | 是否启用多任务训练；为 `true` 时按 `env.tasks` 逐任务独立训练 |
| `log_interval` | 日志打印间隔 |
| `save_interval` | checkpoint 保存间隔 |
| `eval_interval` | 每隔多少 `step` 评估一次，必须大于 0 |
| `eval_episodes` | 每次评估跑几个 `episode`，必须大于 0 |
| `record_interval` | metrics / csv / 曲线图采样间隔 |
| `device` | `cpu` / `cuda` |

## Env

| Field | Description |
| --- | --- |
| `type` | 环境后端：`dmcontrol` / `isaac` |
| `name` | 任务别名 |
| `domain_name` | `dm_control` 的 `domain` |
| `task_name` | `dm_control` 的 `task` |
| `tasks` | 多任务列表；每项至少给出 `name`、`domain_name`、`task_name` |
| `observation.type` | 观测类型：`vector` 或 `image` |
| `observation.num_cams` | 图像模式下的相机个数，环境返回 `n` 路图像观测 |
| `action.clip` | 是否裁剪动作到合法范围 |
| `action.normalize` | 策略是否默认输出 `[-1, 1]` |
| `render.enabled` | 是否渲染 |
| `render.save_frames` | 是否保存 `png` 帧 |
| `render.save_video` | 是否保存视频 |
| `render.backend` | `dm_control` 图像渲染后端，可选 `egl` / `osmesa` / `glfw` |
| `render.backend_priority` | 当 `render.backend` 为空时的候选顺序，默认优先 `egl` |
| `render.allow_software_render_fallback` | 是否允许在主渲染后端失败时回退 `osmesa` |
| `isaac.task_name` | Isaac 任务名 |
| `isaac.num_envs` | Isaac 并行环境数量 |
| `isaac.headless` | 是否无窗口运行 |
| `isaac.create_cam` | Isaac 相机创建逻辑预留字段，后续补实现 |

## Policy

| Field | Description |
| --- | --- |
| `type` | 策略文件夹名，例如 `random` / `ppo` / `dreamer` |
| `name` | 策略别名 |
| `checkpoint.load` | 是否加载已有 checkpoint |
| `checkpoint.path` | checkpoint 路径 |
| `checkpoint.save` | 是否保存 checkpoint |
| `dreamerv3.device` | DreamerV3 训练设备，建议生产训练使用 `cuda` |
| `dreamerv3.allow_cpu_fallback` | 当 `device=cuda` 但 CUDA 不可用时，是否允许回退到 `cpu` |
| `random.action_low` | 随机动作下界 |
| `random.action_high` | 随机动作上界 |

## Observation Convention

- 环境侧负责按 `env.observation.type` 提供原始观测：`vector` 返回状态向量，`image` 返回单张图像或多相机图像列表。
- `env.observation.num_cams = n` 表示图像模式下环境向策略提供 `n` 路相机图像。
- 策略拿到图像后，需要自行 `reshape` / `stack` / `permute` 成自己可消费的输入格式。
- `dmcontrol` 暂时不用 `create_cam` 逻辑，相关相机创建约定主要为 Isaac 预留。
- Isaac 的 `create_cam` 具体实现后续再加，现阶段先在 YAML 中保留字段。

## Multi-task Example

```yaml
train:
  seed: 0
  total_steps: 10000
  max_episode_steps: 200
  log_interval: 100
  save_interval: 1000
  eval_interval: 5000
  eval_episodes: 2
  record_interval: 100
  device: cpu
  multi_task: true

env:
  type: dmcontrol
  observation:
    type: image
    num_cams: 1
  action:
    clip: true
    normalize: true
  render:
    enabled: false
    save_frames: false
    save_video: false
  tasks:
    - name: cartpole_swingup
      domain_name: cartpole
      task_name: swingup
    - name: cartpole_balance
      domain_name: cartpole
      task_name: balance
```

- 多任务模式下，`scripts/train.py` 会为 `env.tasks` 中的每个任务分别新建 `env` 和 `policy`。
- 输出目录会变成 `exam/<exam_name>/output/<task_name>/...`。
- 每个任务都从头开始训练，不会继承本次前一个任务的策略权重。

## Note

`scripts/create_exam.py`、`scripts/train.py`、`scripts/run_exam.py` 应默认使用这份结构。
其中 `scripts/create_exam.py` 会强制生成带评估的配置，`scripts/train.py` 会按 `eval_interval` 在训练过程中执行评估并保存 GIF。

旧版 `experiment/runtime` 配置目前只作为兼容读取保留，后续应逐步淘汰。
