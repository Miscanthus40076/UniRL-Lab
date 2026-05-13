# MLP Policy Implementation

This policy uses a simple Multi-Layer Perceptron (MLP) network to map observations to actions, with added Gaussian noise for exploration.

## Iteration History

### v1.0 (2026-05-08)
- Initial implementation of MLP-based policy
- Uses 2-layer network with tanh activation
- Added Gaussian noise for exploration
- Output bounded to [-1, 1] range