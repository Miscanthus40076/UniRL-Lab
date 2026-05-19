# Policy Versioning Convention

当前仓库中的 `policy/` 分成两层：

- `policy/<name>/`
  - 工作区实现，可继续开发
- `policy/versions/<version>/<name>/`
  - 冻结快照，供实验复现使用

## 强制约定

1. 任何已经用于实验的策略，不再原地迭代。
2. 新迭代必须先复制到新的版本目录，再修改。
3. `policy/make_policy.py` 只从 `policy/registry.py` 指向的冻结版本构造策略对象。
4. 新实验应显式在 YAML 中写：

```yaml
policy:
  type: dreamerv3
  version: v1
```

也支持紧凑写法：

```yaml
policy:
  type: dreamerv3@v1
```

## 当前冻结版本

- `random -> policy/versions/v1/random`
- `mlp -> policy/versions/v1/mlp`
- `dreamerv3 -> policy/versions/v1/dreamerv3`
- `operator_token_probe -> policy/versions/v1/operator_token_probe`

## 迭代流程

1. 复制冻结版本目录，例如 `v1 -> v2`
2. 在新目录中修改实现
3. 在 `policy/registry.py` 注册 `v2`
4. 新 exam 显式绑定到 `version: v2`

旧版本不再修改。

