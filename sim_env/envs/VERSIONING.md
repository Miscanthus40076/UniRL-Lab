# Environment Versioning Convention

当前仓库中的 `sim_env/envs/` 分成两层：

- `sim_env/envs/*.py`
  - 工作区实现，可继续开发
- `sim_env/envs/versions/<version>/`
  - 冻结快照，供实验复现使用

## 强制约定

1. 任何已经用于实验的环境实现，不再原地迭代。
2. 新迭代必须先复制到新的版本目录，再修改。
3. `sim_env/envs/make_env.py` 只从 `sim_env/envs/registry.py` 指向的冻结版本构造环境对象。
4. 新实验应显式在 YAML 中写：

```yaml
env:
  type: metaworld
  version: v1
```

## 当前冻结版本

`sim_env/envs/versions/v1/` 当前保存了：

- `dm_control_env.py`
- `ball_in_cup_take_out_env.py`
- `metaworld_env.py`
- `persistent_exploration_env.py`
- `sawyer_peg_insertion_side_sparse_v3.py`
- `sawyer_peg_insertion_side_reverse_v3.py`
- `sawyer_peg_insertion_side_reverse_shaped_v3.py`
- `peg_insert_side_reverse_sparse/`
  - 独立反向拔杆子 sparse 任务包
  - 用 `env.type: peg_insert_side_reverse_sparse` 调用
  - 不依赖 `metaworld_env.py` 的自定义任务映射

## 迭代流程

1. 复制 `versions/v1 -> versions/v2`
2. 修改新版本中的环境代码
3. 在 `sim_env/envs/registry.py` 注册 `v2`
4. 新 exam 显式绑定到 `env.version: v2`

旧版本不再修改。
