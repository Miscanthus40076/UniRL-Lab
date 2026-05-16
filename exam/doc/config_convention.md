# Exam Config Convention

## YAML Template

```yaml
train:
  trainer: online_policy
  seed: 0
  total_steps: 10000
  max_episode_steps: null
  multi_task: false
  bidirectional:
    enabled: false
    stage: actor_value_with_prior
    forward_env: {}
    reverse_env: {}

  log_interval: 100
  save_interval: 1000

  eval_interval: 5000
  eval_episodes: 5
  eval_visualize_episodes: 5
  eval_success_metric: episode_return_positive
  eval_success_threshold: 0.0

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
| `trainer` | 训练入口类型：默认 `online_policy`，双向训练可用 `bidirectional` / `bidreamer` |
| `seed` | 随机种子 |
| `total_steps` | 总训练步数 |
| `max_episode_steps` | 单 `episode` 最大步数，可选 |
| `multi_task` | 是否启用多任务训练；为 `true` 时按 `env.tasks` 逐任务独立训练 |
| `bidirectional.enabled` | 是否启用双向训练模式 |
| `bidirectional.stage` | 双向训练阶段，当前支持 `world_model_only` / `actor_value_no_prior` / `actor_value_with_prior` |
| `bidirectional.forward_env` | 正向训练环境覆盖项；基于顶层 `env` 深拷贝后覆盖 |
| `bidirectional.reverse_env` | 反向训练环境覆盖项；基于顶层 `env` 深拷贝后覆盖 |
| `log_interval` | 日志打印间隔 |
| `save_interval` | checkpoint 保存间隔 |
| `eval_interval` | 每隔多少 `step` 评估一次，必须大于 0 |
| `eval_episodes` | 每次评估跑几个 `episode`，必须大于 0 |
| `eval_visualize_episodes` | 每次评估中实际保存 GIF 的 `episode` 数，必须小于等于 `eval_episodes` |
| `eval_success_metric` | 成功判定方式：`episode_return_positive` / `episode_return_threshold` / `info_success` |
| `eval_success_threshold` | 需要阈值时使用的成功阈值 |
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
- `ball_in_cup` 额外支持一个本地自定义任务：`task_name: release`（或 `take_out`）。
  它会让球初始就在杯子里，目标变成把球从杯子里拿出来，成功时 `info.success = true`。
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

## Bidirectional Example

```yaml
train:
  trainer: bidirectional
  seed: 0
  bidirectional:
    enabled: true
    stage: actor_value_with_prior
    forward_env:
      name: ball_in_cup_catch
      domain_name: ball_in_cup
      task_name: catch
    reverse_env:
      name: ball_in_cup_release
      domain_name: ball_in_cup
      task_name: release

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
    height: 64
    width: 64
    camera_id: 0
    backend_priority: [egl, osmesa]
    allow_software_render_fallback: true

training:
  total_env_steps: 100000
  warmup_env_steps_per_direction: 1000
  batch_size: 16
  seq_len: 16
  eval_interval: 1000
  learning_rate: 3.0e-4
  grad_clip: 100.0
  device: cuda_if_available_else_cpu

replay:
  capacity: 100000
```

- 双向模式下，`scripts/train.py` 会实例化 `Trainer` 子类，而不是走默认的单策略在线更新循环。
- 当前双向模式不支持 `train.multi_task = true`。
- `train.bidirectional.forward_env` / `reverse_env` 只需要写和顶层 `env` 不同的部分；渲染、观测、动作配置会从顶层 `env` 继承。

## Note

`scripts/create_exam.py`、`scripts/train.py`、`scripts/run_exam.py` 应默认使用这份结构。
其中 `scripts/create_exam.py` 会强制生成带评估的配置，并为每个 `exam/<name>/` 自动生成一个 `train.py`。
`scripts/train.py` 现在只负责加载 `exam/<name>/train.py`，后者必须定义一个继承主入口 `scripts/train.py` 中 `BaseExamTrainApp` 的 `ExamTrain` 类，并在类里显式给出训练器构造逻辑。
因此每个 exam 都是自带训练入口的自描述目录，支持直接运行 `python exam/<name>/train.py`，也支持统一入口 `python scripts/train.py <name>`。

旧版 `experiment/runtime` 配置目前只作为兼容读取保留，后续应逐步淘汰。
