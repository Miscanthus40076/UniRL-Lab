# Environment Wrapper Versioning

Wrappers adapt a frozen environment version to a trainer-specific runtime
contract. They are selected from exam YAML and should stay free of algorithm
logic.

Example:

```yaml
wrappers:
  - type: simer_to_embodied
    version: v1
    obs_key: image
```

Current wrappers:

- `simer_to_embodied@v1`: adapts the project `BaseEnv` API to the official
  DreamerV3 `embodied.Env` API.

Use a new wrapper version when reset semantics, observation keys, dtype/shape
rules, action scaling, or terminal/truncation behavior changes.
