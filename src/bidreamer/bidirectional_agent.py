from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from policy.dreamerv3.actor import DreamerActor, DreamerActorConfig
from policy.dreamerv3.heads import TwoHotSymlogHead
from policy.dreamerv3.lambda_return import lambda_return
from policy.dreamerv3.normalization import RunningNormConfig, RunningNormalizer
from policy.dreamerv3.transforms import symexp, symlog, twohot_logprob, twohot_mean
from policy.dreamerv3.value import DreamerValue, DreamerValueConfig

from .direction_world_model import DirectionConditionedWorldModel


class BidirectionalAgent:
    def __init__(
        self,
        model: DirectionConditionedWorldModel,
        learning_rate: float,
        grad_clip: float,
        actor_lr: float = 8e-5,
        value_lr: float = 8e-5,
        actor_grad_clip: float = 100.0,
        value_grad_clip: float = 100.0,
        imagination_horizon: int = 15,
        gamma: float = 0.99,
        lambda_: float = 0.95,
        entropy_coef: float = 1e-3,
        actor_hidden_dim: int = 256,
        actor_num_layers: int = 2,
        value_hidden_dim: int = 256,
        value_num_layers: int = 2,
        min_std: float = 0.1,
        max_std: float = 1.0,
        init_std: float = 1.0,
        imag_last: int = 0,
        use_twohot_value: bool = True,
        use_slow_value: bool = True,
        slow_value_rate: float = 0.02,
        repval_loss: bool = False,
        repval_scale: float = 0.3,
        use_return_norm: bool = True,
        use_advantage_norm: bool = True,
        norm_rate: float = 0.01,
        norm_eps: float = 1e-8,
        latent_prior_enabled: bool = False,
        latent_prior_beta: float = 1.0,
        latent_prior_horizon: int | None = None,
        latent_prior_subgoal_step: int = 1,
    ):
        self.model = model
        self.grad_clip = float(grad_clip)
        self.actor_grad_clip = float(actor_grad_clip)
        self.value_grad_clip = float(value_grad_clip)
        self.imagination_horizon = int(imagination_horizon)
        self.gamma = float(gamma)
        self.lambda_ = float(lambda_)
        self.entropy_coef = float(entropy_coef)
        self.imag_last = int(imag_last)
        self.use_twohot_value = bool(use_twohot_value)
        self.use_slow_value = bool(use_slow_value)
        self.slow_value_rate = float(slow_value_rate)
        self.repval_loss = bool(repval_loss)
        self.repval_scale = float(repval_scale)
        self.use_return_norm = bool(use_return_norm)
        self.use_advantage_norm = bool(use_advantage_norm)
        self.latent_prior_enabled = bool(latent_prior_enabled)
        self.latent_prior_beta = float(latent_prior_beta)
        self.latent_prior_horizon = int(latent_prior_horizon or imagination_horizon)
        self.latent_prior_subgoal_step = int(latent_prior_subgoal_step)
        self.device = next(self.model.parameters()).device

        feat_dim = self.model.feat_dim
        action_dim = int(self.model.config.action_dim)
        actor_cfg = DreamerActorConfig(
            feat_dim=feat_dim,
            action_dim=action_dim,
            hidden_dim=int(actor_hidden_dim),
            num_layers=int(actor_num_layers),
            min_std=float(min_std),
            max_std=float(max_std),
            init_std=float(init_std),
        )
        self.forward_actor = DreamerActor(actor_cfg).to(self.device)
        self.reverse_actor = DreamerActor(actor_cfg).to(self.device)
        self._stabilize_actor_init(self.forward_actor, action_dim)
        self._stabilize_actor_init(self.reverse_actor, action_dim)
        self.forward_value = self._make_value_model(feat_dim, value_hidden_dim, value_num_layers).to(self.device)
        self.reverse_value = self._make_value_model(feat_dim, value_hidden_dim, value_num_layers).to(self.device)
        self.forward_slow_value = (
            self._make_value_model(feat_dim, value_hidden_dim, value_num_layers).to(self.device) if self.use_slow_value else None
        )
        self.reverse_slow_value = (
            self._make_value_model(feat_dim, value_hidden_dim, value_num_layers).to(self.device) if self.use_slow_value else None
        )
        for fast, slow in ((self.forward_value, self.forward_slow_value), (self.reverse_value, self.reverse_slow_value)):
            if slow is not None:
                slow.load_state_dict(fast.state_dict())
                for param in slow.parameters():
                    param.requires_grad_(False)

        self.world_model_optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate))
        self.forward_actor_optimizer = torch.optim.Adam(self.forward_actor.parameters(), lr=float(actor_lr))
        self.reverse_actor_optimizer = torch.optim.Adam(self.reverse_actor.parameters(), lr=float(actor_lr))
        self.forward_value_optimizer = torch.optim.Adam(self.forward_value.parameters(), lr=float(value_lr))
        self.reverse_value_optimizer = torch.optim.Adam(self.reverse_value.parameters(), lr=float(value_lr))

        norm_cfg = RunningNormConfig(rate=float(norm_rate), eps=float(norm_eps))
        self.forward_return_normalizer = RunningNormalizer(norm_cfg)
        self.reverse_return_normalizer = RunningNormalizer(norm_cfg)
        self.forward_advantage_normalizer = RunningNormalizer(norm_cfg)
        self.reverse_advantage_normalizer = RunningNormalizer(norm_cfg)

        self._policy_state = {}
        for direction in ("forward", "reverse"):
            self.reset_policy_state(direction)

    def _stabilize_actor_init(self, actor: DreamerActor, action_dim: int):
        last = actor.net[-1]
        if isinstance(last, torch.nn.Linear):
            with torch.no_grad():
                last.weight[:action_dim].zero_()
                last.bias[:action_dim].zero_()

    def _make_value_model(self, feat_dim: int, hidden_dim: int, num_layers: int):
        if self.use_twohot_value:
            return TwoHotSymlogHead(
                feat_dim=feat_dim,
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
                num_bins=self.model.config.reward_bins,
                low=self.model.config.reward_low,
                high=self.model.config.reward_high,
            )
        return DreamerValue(
            DreamerValueConfig(
                feat_dim=feat_dim,
                hidden_dim=int(hidden_dim),
                num_layers=int(num_layers),
            )
        )

    def _assert_direction(self, batch: dict, expected: str):
        direction = batch.get("direction")
        if direction != expected:
            raise ValueError(f"Expected {expected} batch, got {direction}")

    def _actor_for(self, direction: str) -> DreamerActor:
        if direction == "forward":
            return self.forward_actor
        if direction == "reverse":
            return self.reverse_actor
        raise ValueError(f"Unsupported direction: {direction}")

    def _value_for(self, direction: str):
        return self.forward_value if direction == "forward" else self.reverse_value

    def _slow_value_for(self, direction: str):
        return self.forward_slow_value if direction == "forward" else self.reverse_slow_value

    def _value_optimizer_for(self, direction: str):
        return self.forward_value_optimizer if direction == "forward" else self.reverse_value_optimizer

    def _actor_optimizer_for(self, direction: str):
        return self.forward_actor_optimizer if direction == "forward" else self.reverse_actor_optimizer

    def _return_normalizer_for(self, direction: str) -> RunningNormalizer:
        return self.forward_return_normalizer if direction == "forward" else self.reverse_return_normalizer

    def _advantage_normalizer_for(self, direction: str) -> RunningNormalizer:
        return self.forward_advantage_normalizer if direction == "forward" else self.reverse_advantage_normalizer

    def _sync_slow_value(self, direction: str):
        slow = self._slow_value_for(direction)
        if slow is None:
            return
        fast = self._value_for(direction)
        with torch.no_grad():
            for slow_param, fast_param in zip(slow.parameters(), fast.parameters()):
                slow_param.data.lerp_(fast_param.data, self.slow_value_rate)

    def _head_to_scalar(self, module, feat: torch.Tensor, direction: str, denormalize: bool = True) -> torch.Tensor:
        if isinstance(module, TwoHotSymlogHead):
            logits = module(feat)
            value = symexp(
                twohot_mean(
                    logits,
                    num_bins=self.model.config.reward_bins,
                    low=self.model.config.reward_low,
                    high=self.model.config.reward_high,
                )
            )
        else:
            value = module(feat)
        if denormalize and self.use_return_norm:
            return self._return_normalizer_for(direction).denormalize(value)
        return value

    def _value_forward(self, feat: torch.Tensor, direction: str, denormalize: bool = True) -> torch.Tensor:
        return self._head_to_scalar(self._value_for(direction), feat, direction, denormalize=denormalize)

    def _slow_value_forward(self, feat: torch.Tensor, direction: str, denormalize: bool = True) -> torch.Tensor:
        slow = self._slow_value_for(direction)
        if slow is None:
            return self._value_forward(feat, direction, denormalize=denormalize)
        return self._head_to_scalar(slow, feat, direction, denormalize=denormalize)

    def _value_loss(self, feat: torch.Tensor, target: torch.Tensor, direction: str) -> torch.Tensor:
        value_module = self._value_for(direction)
        target = target.reshape(-1)
        if self.use_return_norm:
            target = self._return_normalizer_for(direction).normalize(target, update=False)
        if isinstance(value_module, TwoHotSymlogHead):
            target = symlog(target)
            logits = value_module(feat)
            return -twohot_logprob(
                logits,
                target,
                num_bins=self.model.config.reward_bins,
                low=self.model.config.reward_low,
                high=self.model.config.reward_high,
            ).mean()
        return F.mse_loss(value_module(feat), target)

    def _set_world_model_trainable(self, trainable: bool):
        for param in self.model.parameters():
            param.requires_grad_(trainable)

    def _posterior_start_state(self, batch: dict[str, torch.Tensor], direction: str) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model.posterior_outputs(batch, direction)
            post = outputs["post"]
        batch_size, seq_len, _ = post["h"].shape
        if self.imag_last > 0:
            steps = min(self.imag_last, seq_len)
            return {
                key: value[:, -steps:].reshape(batch_size * steps, *value.shape[2:]).detach()
                for key, value in post.items()
            }
        return {
            key: value.reshape(batch_size * seq_len, *value.shape[2:]).detach()
            for key, value in post.items()
        }

    def _discount_weights(self, continues: torch.Tensor) -> torch.Tensor:
        discounts = self.gamma * continues.detach()
        prefix = torch.ones(1, discounts.shape[1], dtype=discounts.dtype, device=discounts.device)
        if discounts.shape[0] == 1:
            return prefix
        return torch.cat([prefix, torch.cumprod(discounts[:-1], dim=0)], dim=0)

    def _imagine_rollout(self, direction: str, start_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        actor = self._actor_for(direction)
        state = {key: value for key, value in start_state.items()}
        feats = []
        rewards = []
        continues = []
        log_probs = []
        entropies = []
        states = []
        actions = []
        for _ in range(self.imagination_horizon):
            feat = self.model.rssm.get_feat(state)
            action, log_prob, entropy = actor.sample(feat)
            reward = self.model.reward_prediction(feat, direction)
            continue_prob = torch.sigmoid(self.model.continue_prediction(feat, direction))
            state = self.model.rssm.imagine_step(state, action)
            feats.append(feat)
            actions.append(action)
            rewards.append(reward)
            continues.append(continue_prob)
            log_probs.append(log_prob)
            entropies.append(entropy)
            states.append({key: value for key, value in state.items()})
        return {
            "feats": torch.stack(feats, dim=0),
            "actions": torch.stack(actions, dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "continues": torch.stack(continues, dim=0),
            "log_probs": torch.stack(log_probs, dim=0),
            "entropies": torch.stack(entropies, dim=0),
            "states": {key: torch.stack([state[key] for state in states], dim=0) for key in states[0]},
        }

    def _replay_value_loss(self, batch: dict[str, torch.Tensor], direction: str) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model.posterior_outputs(batch, direction)
            feat = outputs["feat"]
        flat_feat = feat.reshape(-1, feat.shape[-1])
        slow_values = self._slow_value_forward(flat_feat, direction).view(feat.shape[0], feat.shape[1])
        current_values = self._value_forward(flat_feat, direction).view(feat.shape[0], feat.shape[1])
        bootstrap = slow_values[:, -1]
        replay_returns = lambda_return(
            rewards=batch["reward"].transpose(0, 1),
            values=torch.cat([slow_values.transpose(0, 1), bootstrap.unsqueeze(0)], dim=0),
            continues=(1.0 - batch["done"]).transpose(0, 1),
            gamma=self.gamma,
            lambda_=self.lambda_,
        ).transpose(0, 1)
        repval_loss = self._value_loss(flat_feat, replay_returns.detach(), direction)
        slow_reg = F.mse_loss(current_values, slow_values.detach())
        return repval_loss, slow_reg

    def train_world_model(self, forward_batch: dict, reverse_batch: dict) -> dict[str, float | list[float]]:
        self._assert_direction(forward_batch, "forward")
        self._assert_direction(reverse_batch, "reverse")
        self.model.train()
        mixed = self.model.mixed_loss(forward_batch, reverse_batch)
        loss = mixed["mixed_total_loss"]
        if torch.isnan(loss).any():
            raise RuntimeError("world model loss is NaN")
        self.world_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.world_model_optimizer.step()
        metrics = {}
        for key, value in mixed.items():
            if isinstance(value, dict):
                continue
            metrics[key] = float(value.detach().cpu()) if torch.is_tensor(value) else value
        return metrics

    def _train_actor_value(self, batch: dict[str, torch.Tensor], direction: str) -> dict[str, float]:
        self._assert_direction(batch, direction)
        actor = self._actor_for(direction)
        value = self._value_for(direction)
        actor_opt = self._actor_optimizer_for(direction)
        value_opt = self._value_optimizer_for(direction)
        actor.train()
        value.train()
        start_state = self._posterior_start_state(batch, direction)
        self._set_world_model_trainable(False)

        imagined = self._imagine_rollout(direction, start_state)
        feats = imagined["feats"]
        rewards = imagined["rewards"]
        continues = imagined["continues"]
        entropies = imagined["entropies"]
        log_probs = imagined["log_probs"]
        flat_feats = feats.reshape(-1, feats.shape[-1])
        values = self._value_forward(flat_feats, direction).view(feats.shape[0], feats.shape[1])
        last_state = {key: value[-1] for key, value in imagined["states"].items()}
        bootstrap = self._slow_value_forward(self.model.rssm.get_feat(last_state), direction).unsqueeze(0)
        all_values = torch.cat([values, bootstrap], dim=0)
        returns = lambda_return(
            rewards=rewards,
            values=all_values,
            continues=continues,
            gamma=self.gamma,
            lambda_=self.lambda_,
        )
        if self.use_return_norm:
            self._return_normalizer_for(direction).update(returns.detach())
        weights = self._discount_weights(continues)
        advantage = returns.detach() - values.detach()
        if self.use_advantage_norm:
            norm_advantage = self._advantage_normalizer_for(direction).normalize(advantage, update=True)
        else:
            norm_advantage = advantage
        actor_loss = -(weights * (log_probs * norm_advantage + self.entropy_coef * entropies)).mean()
        if torch.isnan(actor_loss).any():
            raise RuntimeError(f"{direction} actor loss is NaN")
        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), self.actor_grad_clip)
        actor_opt.step()

        value_loss = self._value_loss(flat_feats.detach(), returns.detach(), direction)
        repval_loss = torch.zeros((), device=self.device)
        slowreg_loss = torch.zeros((), device=self.device)
        if self.repval_loss:
            repval_loss, slowreg_loss = self._replay_value_loss(batch, direction)
            value_loss = value_loss + self.repval_scale * repval_loss + self.slow_value_rate * slowreg_loss
        if torch.isnan(value_loss).any():
            raise RuntimeError(f"{direction} value loss is NaN")
        value_opt.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), self.value_grad_clip)
        value_opt.step()
        self._sync_slow_value(direction)
        self._set_world_model_trainable(True)

        value_pred = self._value_forward(flat_feats.detach(), direction).view(feats.shape[0], feats.shape[1])
        prefix = f"{direction}_"
        return {
            f"{prefix}actor_loss": float(actor_loss.detach().cpu()),
            f"{prefix}value_loss": float(value_loss.detach().cpu()),
            f"{prefix}repval_loss": float(repval_loss.detach().cpu()),
            f"{prefix}slowreg_loss": float(slowreg_loss.detach().cpu()),
            f"{prefix}imagined_return": float(returns.detach().mean().cpu()),
            f"{prefix}value_pred": float(value_pred.detach().mean().cpu()),
            f"{prefix}entropy": float(entropies.detach().mean().cpu()),
            f"{prefix}action_abs_mean": float(imagined["actions"].detach().abs().mean().cpu()),
            f"{prefix}action_abs_max": float(imagined["actions"].detach().abs().max().cpu()),
        }

    def train_forward_actor_value(self, forward_batch: dict, reverse_latent_memory=None) -> dict[str, float]:
        if reverse_latent_memory is not None and not self.latent_prior_enabled:
            raise ValueError("reverse_latent_memory was provided but latent_prior is disabled in this stage")
        self._assert_direction(forward_batch, "forward")
        actor = self.forward_actor
        value = self.forward_value
        actor.train()
        value.train()
        start_state = self._posterior_start_state(forward_batch, "forward")
        self._set_world_model_trainable(False)

        imagined = self._imagine_rollout("forward", start_state)
        feats = imagined["feats"]
        rewards = imagined["rewards"]
        continues = imagined["continues"]
        entropies = imagined["entropies"]
        log_probs = imagined["log_probs"]
        flat_feats = feats.reshape(-1, feats.shape[-1])
        values = self._value_forward(flat_feats, "forward").view(feats.shape[0], feats.shape[1])
        last_state = {key: value[-1] for key, value in imagined["states"].items()}
        bootstrap = self._slow_value_forward(self.model.rssm.get_feat(last_state), "forward").unsqueeze(0)
        all_values = torch.cat([values, bootstrap], dim=0)
        returns = lambda_return(
            rewards=rewards,
            values=all_values,
            continues=continues,
            gamma=self.gamma,
            lambda_=self.lambda_,
        )
        if self.use_return_norm:
            self.forward_return_normalizer.update(returns.detach())
        weights = self._discount_weights(continues)
        advantage = returns.detach() - values.detach()
        if self.use_advantage_norm:
            norm_advantage = self.forward_advantage_normalizer.normalize(advantage, update=True)
        else:
            norm_advantage = advantage
        actor_loss = -(weights * (log_probs * norm_advantage + self.entropy_coef * entropies)).mean()

        latent_prior_loss = torch.zeros((), device=self.device)
        latent_prior_raw_loss = torch.zeros((), device=self.device)
        mean_prior_weight = torch.zeros((), device=self.device)
        latent_subgoal_distance = torch.zeros((), device=self.device)
        latent_prior_active = 0.0
        if self.latent_prior_enabled and reverse_latent_memory is not None:
            latent_prior_raw_loss, mean_prior_weight, latent_subgoal_distance = self._compute_forward_latent_prior(
                start_state=start_state,
                imagined_states=imagined["states"],
                reverse_latent_memory=reverse_latent_memory,
            )
            latent_prior_loss = self.latent_prior_beta * latent_prior_raw_loss
            actor_loss = actor_loss + latent_prior_loss
            latent_prior_active = 1.0

        if torch.isnan(actor_loss).any():
            raise RuntimeError("forward actor loss is NaN")
        self.forward_actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), self.actor_grad_clip)
        self.forward_actor_optimizer.step()

        value_loss = self._value_loss(flat_feats.detach(), returns.detach(), "forward")
        repval_loss = torch.zeros((), device=self.device)
        slowreg_loss = torch.zeros((), device=self.device)
        if self.repval_loss:
            repval_loss, slowreg_loss = self._replay_value_loss(forward_batch, "forward")
            value_loss = value_loss + self.repval_scale * repval_loss + self.slow_value_rate * slowreg_loss
        if torch.isnan(value_loss).any():
            raise RuntimeError("forward value loss is NaN")
        self.forward_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), self.value_grad_clip)
        self.forward_value_optimizer.step()
        self._sync_slow_value("forward")
        self._set_world_model_trainable(True)

        value_pred = self._value_forward(flat_feats.detach(), "forward").view(feats.shape[0], feats.shape[1])
        return {
            "forward_actor_loss": float(actor_loss.detach().cpu()),
            "forward_value_loss": float(value_loss.detach().cpu()),
            "forward_repval_loss": float(repval_loss.detach().cpu()),
            "forward_slowreg_loss": float(slowreg_loss.detach().cpu()),
            "forward_imagined_return": float(returns.detach().mean().cpu()),
            "forward_value_pred": float(value_pred.detach().mean().cpu()),
            "forward_entropy": float(entropies.detach().mean().cpu()),
            "forward_action_abs_mean": float(imagined["actions"].detach().abs().mean().cpu()),
            "forward_action_abs_max": float(imagined["actions"].detach().abs().max().cpu()),
            "latent_prior_loss": float(latent_prior_loss.detach().cpu()),
            "latent_prior_raw_loss": float(latent_prior_raw_loss.detach().cpu()),
            "mean_prior_weight": float(mean_prior_weight.detach().cpu()),
            "latent_subgoal_distance": float(latent_subgoal_distance.detach().cpu()),
            "latent_prior_beta": float(self.latent_prior_beta if latent_prior_active else 0.0),
            "latent_prior_active": float(latent_prior_active),
        }

    def train_reverse_actor_value(self, reverse_batch: dict) -> dict[str, float]:
        return self._train_actor_value(reverse_batch, "reverse")

    def _compute_forward_latent_prior(self, start_state: dict[str, torch.Tensor], imagined_states: dict[str, torch.Tensor], reverse_latent_memory) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start_feats = self.model.extract_feat(start_state, deterministic=True).detach()
        horizon_idx = min(max(self.latent_prior_horizon, 1), imagined_states["h"].shape[0]) - 1
        imagined_state_h = {key: value[horizon_idx] for key, value in imagined_states.items()}
        imagined_feat_h = self.model.extract_feat(imagined_state_h, deterministic=True)
        target_rows = []
        distance_rows = []
        weight_rows = []
        for feat in start_feats:
            nearest = reverse_latent_memory.nearest(feat, k=1)[0]
            subgoal = reverse_latent_memory.get_subgoal(nearest, subgoal_step=self.latent_prior_subgoal_step)
            target_rows.append(subgoal["feature"].to(self.device).detach())
            distance = float(nearest.distance)
            distance_rows.append(distance)
            weight_rows.append(float(np.exp(-distance)))
        targets = torch.stack(target_rows, dim=0)
        distances = torch.as_tensor(distance_rows, dtype=torch.float32, device=self.device)
        prior_weight = torch.as_tensor(weight_rows, dtype=torch.float32, device=self.device)
        mse_per_item = (imagined_feat_h - targets).pow(2).mean(dim=-1)
        raw_loss = (prior_weight.detach() * mse_per_item).mean()
        return raw_loss, prior_weight.mean(), distances.mean()

    def reset_policy_state(self, direction: str):
        self._policy_state[direction] = {
            "latent": self.model.rssm.init_state(1, self.device),
            "prev_action": torch.zeros(1, self.model.config.action_dim, dtype=torch.float32, device=self.device),
            "is_first": True,
        }

    def snapshot_policy_state(self) -> dict[str, dict[str, object]]:
        snapshot = {}
        for direction, payload in self._policy_state.items():
            snapshot[direction] = {
                "latent": {key: value.detach().clone() for key, value in payload["latent"].items()},
                "prev_action": payload["prev_action"].detach().clone(),
                "is_first": bool(payload["is_first"]),
            }
        return snapshot

    def restore_policy_state(self, snapshot: dict[str, dict[str, object]]):
        restored = {}
        for direction, payload in snapshot.items():
            restored[direction] = {
                "latent": {key: value.to(self.device) for key, value in payload["latent"].items()},
                "prev_action": payload["prev_action"].to(self.device),
                "is_first": bool(payload["is_first"]),
            }
        self._policy_state = restored

    def act(self, obs, direction: str, deterministic: bool = False) -> np.ndarray:
        state = self._policy_state[direction]
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            embed = self.model.encoder(obs_t)
            latent, _ = self.model.rssm.observe_step(
                state=state["latent"],
                embed=embed,
                action=state["prev_action"],
                is_first=torch.as_tensor([float(state["is_first"])], dtype=torch.float32, device=self.device),
            )
            feat = self.model.rssm.get_feat(latent)
            actor = self._actor_for(direction)
            action = actor.mode(feat) if deterministic else actor.sample(feat)[0]
        state["latent"] = {key: value.detach() for key, value in latent.items()}
        state["prev_action"] = action.detach()
        state["is_first"] = False
        out = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if not np.all(np.isfinite(out)):
            raise RuntimeError(f"{direction} actor produced non-finite action")
        return out

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "world_model_optimizer_state_dict": self.world_model_optimizer.state_dict(),
                "forward_actor_state_dict": self.forward_actor.state_dict(),
                "reverse_actor_state_dict": self.reverse_actor.state_dict(),
                "forward_value_state_dict": self.forward_value.state_dict(),
                "reverse_value_state_dict": self.reverse_value.state_dict(),
                "forward_slow_value_state_dict": None if self.forward_slow_value is None else self.forward_slow_value.state_dict(),
                "reverse_slow_value_state_dict": None if self.reverse_slow_value is None else self.reverse_slow_value.state_dict(),
                "forward_actor_optimizer_state_dict": self.forward_actor_optimizer.state_dict(),
                "reverse_actor_optimizer_state_dict": self.reverse_actor_optimizer.state_dict(),
                "forward_value_optimizer_state_dict": self.forward_value_optimizer.state_dict(),
                "reverse_value_optimizer_state_dict": self.reverse_value_optimizer.state_dict(),
                "forward_return_normalizer": self.forward_return_normalizer.state_dict(),
                "reverse_return_normalizer": self.reverse_return_normalizer.state_dict(),
                "forward_advantage_normalizer": self.forward_advantage_normalizer.state_dict(),
                "reverse_advantage_normalizer": self.reverse_advantage_normalizer.state_dict(),
                "latent_prior_enabled": self.latent_prior_enabled,
            },
            path,
        )
