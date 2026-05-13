import numpy as np


class MLPNetwork:
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        # Simple 2-layer MLP
        # TODO: 如果该网络承担训练职责，需要补 backward/optimizer/checkpoint；当前只有随机初始化前向推理。
        self.W1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    def forward(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)  # Add batch dimension

        h = np.tanh(x @ self.W1 + self.b1)
        output = np.tanh(h @ self.W2 + self.b2)  # Use tanh to bound outputs to [-1, 1]

        # If input was 1D, return 1D
        if output.shape[0] == 1:
            return output[0]
        return output
