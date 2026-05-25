from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(slots=True)
class RSSMConfig:
    action_dim: int = 4
    embed_dim: int = 128
    deter_dim: int = 128
    stoch_dim: int = 32
    stoch_classes: int = 32
    hidden_dim: int = 256
    unimix: float = 0.01


class RSSM(nn.Module):
    def __init__(self, config: RSSMConfig):
        super().__init__()
        self.config = config
        stoch_size = config.stoch_dim * config.stoch_classes
        self.gru = nn.GRUCell(stoch_size + config.action_dim, config.deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(config.deter_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, stoch_size),
        )
        self.post_net = nn.Sequential(
            nn.Linear(config.deter_dim + config.embed_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, stoch_size),
        )

    def init_state(self, batch_size: int, device=None) -> dict[str, torch.Tensor]:
        device = device or next(self.parameters()).device
        h = torch.zeros(batch_size, self.config.deter_dim, device=device)
        logits = torch.zeros(batch_size, self.config.stoch_dim, self.config.stoch_classes, device=device)
        probs = torch.full_like(logits, 1.0 / self.config.stoch_classes)
        z = probs
        return {"h": h, "z": z, "logits": logits, "probs": probs}

    def _stack(self, states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {key: torch.stack([state[key] for state in states], dim=1) for key in states[0]}

    def _reshape_logits(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.view(raw.shape[0], self.config.stoch_dim, self.config.stoch_classes)

    def _apply_unimix(self, probs: torch.Tensor) -> torch.Tensor:
        if self.config.unimix <= 0.0:
            return probs
        uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
        return (1.0 - self.config.unimix) * probs + self.config.unimix * uniform

    def _dist(self, logits: torch.Tensor) -> torch.distributions.OneHotCategoricalStraightThrough:
        probs = torch.softmax(logits, dim=-1)
        probs = self._apply_unimix(probs)
        return torch.distributions.OneHotCategoricalStraightThrough(probs=probs)

    def _sample(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self._dist(logits)
        z = dist.rsample()
        probs = dist.probs
        return z, probs

    def get_feat(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        z = state["z"].reshape(*state["z"].shape[:-2], -1)
        return torch.cat([state["h"], z], dim=-1)

    def imagine_step(self, state: dict[str, torch.Tensor], action: torch.Tensor) -> dict[str, torch.Tensor]:
        z = state["z"].reshape(state["z"].shape[0], -1)
        gru_input = torch.cat([z, action], dim=-1)
        h = self.gru(gru_input, state["h"])
        logits = self._reshape_logits(self.prior_net(h))
        z, probs = self._sample(logits)
        return {"h": h, "z": z, "logits": logits, "probs": probs}

    def observe_step(
        self,
        state: dict[str, torch.Tensor],
        embed: torch.Tensor,
        action: torch.Tensor,
        is_first: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size = embed.shape[0]
        if is_first is not None:
            keep = (1.0 - is_first.float()).view(batch_size, 1, 1)
            keep_h = keep.view(batch_size, 1)
            state = {
                "h": state["h"] * keep_h,
                "z": state["z"] * keep,
                "logits": state["logits"] * keep,
                "probs": state["probs"] * keep,
            }

        prior_state = self.imagine_step(state, action)
        post_input = torch.cat([prior_state["h"], embed], dim=-1)
        logits = self._reshape_logits(self.post_net(post_input))
        z, probs = self._sample(logits)
        post_state = {
            "h": prior_state["h"],
            "z": z,
            "logits": logits,
            "probs": probs,
        }
        return post_state, prior_state

    def observe(
        self,
        embed_seq: torch.Tensor,
        action_seq: torch.Tensor,
        is_first_seq: torch.Tensor | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = embed_seq.shape
        state = self.init_state(batch_size, embed_seq.device)
        post_states = []
        prior_states = []
        for t in range(seq_len):
            is_first = is_first_seq[:, t] if is_first_seq is not None else None
            state, prior_state = self.observe_step(
                state=state,
                embed=embed_seq[:, t],
                action=action_seq[:, t],
                is_first=is_first,
            )
            post_states.append(state)
            prior_states.append(prior_state)
        return self._stack(post_states), self._stack(prior_states)

    def imagine(self, action_seq: torch.Tensor, init_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        state = {key: value for key, value in init_state.items()}
        states = []
        for t in range(action_seq.shape[1]):
            state = self.imagine_step(state, action_seq[:, t])
            states.append(state)
        return self._stack(states)
