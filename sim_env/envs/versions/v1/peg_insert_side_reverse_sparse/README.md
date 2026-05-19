# Peg Insert Side Reverse Sparse

This folder is a standalone reverse peg insertion task package.

It exists so reverse "unplug the peg" experiments do not depend on the shared
forward MetaWorld wrapper or its custom task registry. New reverse-task
iterations should copy this folder into a new environment version before
modification.

Task semantics:

- Underlying simulator: MetaWorld `peg-insert-side-v3`
- Initial state: peg starts near the inserted pose
- Reward: binary sparse reward
- Success: peg has been unplugged, placed on the table, and released from the gripper

Use from an exam:

```yaml
env:
  type: peg_insert_side_reverse_sparse
  version: v1
```

This package intentionally does not import `sim_env.envs.metaworld_env` or the
forward sparse peg task.
