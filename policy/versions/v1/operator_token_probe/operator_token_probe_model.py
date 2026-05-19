from __future__ import annotations

import torch
from torch import nn

from policy.versions.v1.dreamerv3.operator_tokens import (
    InverseActionProbe,
    NoTokenEffectModel,
    NoTokenInverseActionProbe,
    OperatorEffectModel,
    OperatorTokenConfig,
    OperatorTokenPrior,
    OperatorTokenizerPosterior,
)


class OperatorTokenProbeModel(nn.Module):
    def __init__(self, feat_dim: int, action_dim: int, config: OperatorTokenConfig):
        super().__init__()
        self.config = config
        self.feat_dim = int(feat_dim)
        self.action_dim = int(action_dim)
        self.token_embedding = nn.Embedding(config.num_tokens, config.token_dim)
        self.posterior = OperatorTokenizerPosterior(self.feat_dim, self.action_dim, config)
        self.prior = OperatorTokenPrior(self.feat_dim, self.action_dim, config)
        self.effect = OperatorEffectModel(self.feat_dim, self.action_dim, config.token_dim, config.hidden_dim)
        self.effect_no_token = NoTokenEffectModel(self.feat_dim, self.action_dim, config.hidden_dim)
        self.inverse = InverseActionProbe(self.feat_dim, self.action_dim, config.token_dim, config.hidden_dim)
        self.inverse_no_token = NoTokenInverseActionProbe(self.feat_dim, self.action_dim, config.hidden_dim)

    def _prepare_inputs(self, batch: dict) -> tuple[torch.Tensor, ...]:
        x_t = batch["x_t"]
        x_tp1 = batch["x_tp1"]
        delta_x = batch["delta_x"]
        action_t = batch["action_t"]
        if self.config.detach_features:
            x_t = x_t.detach()
            x_tp1 = x_tp1.detach()
            delta_x = delta_x.detach()
            action_t = action_t.detach()
        return x_t, x_tp1, delta_x, action_t

    def forward(self, batch: dict) -> dict:
        x_t, x_tp1, delta_x, action_t = self._prepare_inputs(batch)
        posterior = self.posterior(x_t, x_tp1, delta_x, action_t)
        token_embedding = posterior["token_onehot_st"] @ self.token_embedding.weight
        prior_logits = self.prior(x_t, action_t)
        pred_delta_x = self.effect(x_t, action_t, token_embedding)
        pred_delta_x_no_token = self.effect_no_token(x_t, action_t)
        pred_action = self.inverse(x_t, delta_x, token_embedding)
        pred_action_no_token = self.inverse_no_token(x_t, delta_x)
        return {
            **posterior,
            "prior_logits": prior_logits,
            "token_embedding": token_embedding,
            "pred_delta_x": pred_delta_x,
            "pred_delta_x_no_token": pred_delta_x_no_token,
            "pred_action": pred_action,
            "pred_action_no_token": pred_action_no_token,
        }
