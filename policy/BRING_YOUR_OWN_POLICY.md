# Bring Your Own Policy

This guide explains how to integrate a custom policy implementation into this project so it can be used with the existing training, evaluation, and logging tools.

The project does not use a plugin or package registry system. Instead, a policy is added as a folder under `policy/`, wired into `policy/make_policy.py`, and then consumed by `scripts/train.py` and `scripts/run_exam.py`.

## Step 1: Create a Policy Folder

Create a dedicated folder under `policy/` for your policy:

```text
policy/
├── make_policy.py
├── base_policy.py
├── my_policy/
│   ├── __init__.py
│   ├── README.md
│   ├── ARCHITECTURE.md
│   └── my_policy_policy_impl.py
```

Recommended files:

- `__init__.py` exports the public policy class
- `README.md` describes what the policy is for
- `ARCHITECTURE.md` records the model structure and iteration history
- `*_policy_impl.py` contains the policy wrapper that the runner instantiates

If your policy has more than one layer of responsibility, split it further:

- `*_policy_impl.py` for the runner-facing wrapper
- `*_agent.py` for training and optimization logic
- `*_model.py` for network assembly and architecture config

## Step 2: Implement the Policy Contract

Every policy must implement the shared runner contract from `policy/base_policy.py`:

```python
action = policy.act(obs)
```

Optional methods are supported and commonly used:

- `policy.update(step_batch)`
- `policy.reset()`
- `policy.save(path)`
- `policy.load(path)`

The current runner expects:

- `act(obs)` returns one action for the current observation
- `update(step_batch)` returns a JSON-serializable metrics payload if the policy learns online
- `reset()` clears episode state such as latent variables or recurrent memory

## Step 3: Decide the Observation Contract

Policies are responsible for preprocessing observations themselves.

Current supported patterns:

- vector observations
- image observations
- multi-camera image observations

The policy wrapper should:

- accept the raw observation returned by the environment
- normalize or reshape it inside the policy
- keep environment wrappers free of policy-specific preprocessing

For Dreamer-style or other latent policies, the wrapper usually also needs to:

- infer observation shape once during initialization
- maintain internal latent state across steps
- reset that state at episode boundaries

## Step 4: Define Your Policy Wrapper

The runner-facing class should stay small and explicit.

Typical responsibilities:

- convert raw observations into model inputs
- manage episode state
- call the agent/model to produce actions
- collect step transitions into an internal buffer if training is online
- return metrics in the project’s unified JSON format

Suggested pattern:

```python
class MyPolicyFolder(BasePolicy):
    def act(self, obs):
        ...

    def update(self, step_batch):
        ...
        return {
            "schema": "policy_metrics/v1",
            "scalars": {...},
            "metadata": {...},
        }

    def reset(self):
        ...
```

## Step 5: Wire the Policy Into the Factory

Add a branch in `policy/make_policy.py` so the runner can create your policy from config:

```python
if policy_type == "my_policy":
    return MyPolicyFolder(...)
```

The factory should only do orchestration:

- read policy config
- pass in environment-derived dimensions or observation example data if needed
- return the policy instance

Do not embed algorithm logic in the factory.

## Step 6: Add Config Under `policy`

Policy-specific hyperparameters should live under the `policy` section in the exam config:

```yaml
policy:
  type: my_policy
  name: my_policy_baseline

  checkpoint:
    load: false
    path: null
    save: false

  my_policy:
    device: cpu
    hidden_dim: 256
    seq_len: 32
```

Keep algorithm-specific values inside the policy’s own subkey.

Avoid putting training internals at the top level of `train:` unless they truly belong to the runner.

## Step 7: Return Unified Metrics

If your policy trains online, `update()` should return a unified payload:

```python
{
    "schema": "policy_metrics/v1",
    "scalars": {
        "model_loss": 1.23,
        "actor_loss": 0.45,
        "value_loss": 0.31,
    },
    "metadata": {
        "replay_size": 1000,
        "batch_size": 16,
    },
}
```

Rules:

- `schema` must stay `policy_metrics/v1`
- `scalars` must contain only scalar numbers
- `metadata` may contain extra JSON-serializable debug information
- the runner will write the full payload to `policy_metrics.jsonl`
- the runner will draw plots from the same JSONL source

This keeps policy internals isolated from the plotting code.

## Step 8: Add Documentation

Every policy folder should include:

- `README.md`
- `ARCHITECTURE.md`

`README.md` should cover:

- what the policy does
- supported observation types
- supported action types
- current implementation status
- known limitations

`ARCHITECTURE.md` should record:

- model pipeline
- module boundaries
- hidden sizes / latent sizes
- input and output shapes
- major architecture changes over time

## Step 9: Test the Integration

At minimum, verify the following:

- `policy.make_policy()` can create your policy from config
- `policy.act(obs)` works with the configured observation type
- `policy.update(step_batch)` returns the unified metrics payload if training is enabled
- `scripts/train.py` can run without policy-specific branches
- `scripts/run_exam.py` can evaluate the policy and save outputs

Good smoke tests:

```bash
python -m py_compile policy/make_policy.py scripts/train.py scripts/run_exam.py
```

and a short training run with a minimal exam config.

## Example Policy Flow

For a latent policy such as DreamerV3, the practical flow is:

```text
raw obs -> policy wrapper -> latent/state update -> action
step transition -> policy internal replay -> update() -> unified metrics JSON
runner -> writes JSONL + CSV -> plotting from JSONL
```

## Current Project Rules

- Keep environment wrappers minimal
- Keep policy preprocessing inside the policy
- Keep training-side plotting generic
- Keep algorithm-specific losses and statistics inside the policy payload
- Keep the runner independent from a specific policy family

## Existing Reference Implementations

You can inspect these folders for working examples:

- `policy/random/`
- `policy/mlp/`
- `policy/dreamerv3/`

The DreamerV3 wrapper is the most complete example of a learning policy with internal replay, training, checkpointing, and unified metrics output.

