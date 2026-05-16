from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .actor import DreamerActor, DreamerActorConfig
from .dreamerv3_model import DreamerAuxConfig, DreamerObservationSpec, DreamerV3ModelConfig
from .imagination import imagine_rollout
from .lambda_return import lambda_return
from .losses import WorldModelLossConfig
from .heads import TwoHotSymlogHead
from .normalization import RunningNormConfig, RunningNormalizer
from .transforms import symlog, symexp, twohot_logprob, twohot_mean
from .value import DreamerValue, DreamerValueConfig
from .world_model import DreamerV3WorldModel, DreamerV3WorldModelConfig

if TYPE_CHECKING:
    from .replay_buffer import EpisodeReplayBuffer


class DreamerV3Agent:
    def __init__(self, config: DreamerV3ModelConfig, loss_config: WorldModelLossConfig | None = None):
        self.config = config
        self.loss_config = loss_config or WorldModelLossConfig(
            free_nats=config.free_nats,
            kl_balance=config.kl_balance,
            use_symlog_obs=config.use_symlog_obs,
            use_symlog_reward=config.use_symlog_reward,
            use_twohot_reward=config.use_twohot_reward,
            twohot_bins=config.twohot_bins,
            twohot_low=config.twohot_low,
            twohot_high=config.twohot_high,
            context_update_penalty=(config.thick_context.update_penalty if config.thick_context.enabled else 0.0),
        )
        self.device = torch.device(config.device)

        wm_config = DreamerV3WorldModelConfig(
            obs_dim=config.observation.obs_dim,
            obs_shape=config.observation.obs_shape,
            action_dim=config.action_dim,
            encoder_type=config.encoder_type,
            embed_dim=config.embed_dim,
            deter_dim=config.deter_dim,
            stoch_dim=config.stoch_dim,
            stoch_classes=config.stoch_classes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            contact_num_classes=config.aux.contact_num_classes,
            predict_grasp=config.aux.predict_grasp,
            use_symlog_obs=config.use_symlog_obs,
            use_symlog_reward=config.use_symlog_reward,
            use_twohot_reward=config.use_twohot_reward,
            reward_bins=config.twohot_bins,
            reward_low=config.twohot_low,
            reward_high=config.twohot_high,
            rssm_unimix=config.rssm_unimix,
            thick_context=config.thick_context,
        )
        self.world_model = DreamerV3WorldModel(wm_config).to(self.device)
        feat_dim = config.augmented_feat_dim
        self.actor = DreamerActor(
            DreamerActorConfig(
                feat_dim=feat_dim,
                action_dim=config.action_dim,
                hidden_dim=config.actor_hidden_dim,
                num_layers=config.actor_num_layers,
                min_std=config.min_std,
                max_std=config.max_std,
                init_std=config.init_std,
            )
        ).to(self.device)
        self.value = self._make_value_model(feat_dim).to(self.device)
        self.slow_value = self._make_value_model(feat_dim).to(self.device) if config.use_slow_value else None
        if self.slow_value is not None:
            self.slow_value.load_state_dict(self.value.state_dict())
            for param in self.slow_value.parameters():
                param.requires_grad_(False)
        self.world_model_optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=config.world_model_lr,
        )
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.value_lr)
        self.latent_state = self.world_model.rssm.init_state(1, self.device)
        self.latent_context = self.world_model.initial_context(1, self.device)
        norm_cfg = RunningNormConfig(rate=config.norm_rate, eps=config.norm_eps)
        self.return_normalizer = RunningNormalizer(norm_cfg)
        self.advantage_normalizer = RunningNormalizer(norm_cfg)
        self.last_context_metrics = self._default_context_metrics()
        self._online_batch_size = 16
        self._online_seq_len = 32
        self._online_warmup_steps = 1000
        self._online_train_every = 1
        self._online_train_ratio = 1.0
        self._online_max_updates_per_step = 0
        self._online_train_calls = 0
        self._online_train_budget = 0.0

    def _default_context_metrics(self) -> dict[str, float]:
        if not self.config.thick_context.enabled:
            return {}
        return {
            "context_gate": 0.0,
            "context_delta_norm": 0.0,
            "context_norm": 0.0,
        }

    def _context_metrics_from_details(self, details: dict | None) -> dict[str, float]:
        if not self.config.thick_context.enabled or not details or details.get("context") is None:
            return {}
        context = details["context"]
        gate = details["context_gate"]
        delta_norm = details["context_delta_norm"]
        return {
            "context_gate": float(gate.detach().mean().cpu()),
            "context_delta_norm": float(delta_norm.detach().mean().cpu()),
            "context_norm": float(torch.linalg.vector_norm(context.detach(), dim=-1).mean().cpu()),
        }

    def get_context_diagnostics(self) -> dict[str, float]:
        return dict(self.last_context_metrics)

    def configure_online_update(
        self,
        batch_size: int,
        seq_len: int,
        warmup_steps: int,
        train_every: int = 1,
        train_ratio: float = 1.0,
        max_updates_per_step: int = 0,
    ):
        self._online_batch_size = int(batch_size)
        self._online_seq_len = int(seq_len)
        self._online_warmup_steps = int(warmup_steps)
        self._online_train_every = max(1, int(train_every))
        self._online_train_ratio = float(train_ratio)
        self._online_max_updates_per_step = int(max_updates_per_step)
        self._online_train_calls = 0
        self._online_train_budget = 0.0

    def reset_online_update_state(self):
        self._online_train_calls = 0
        self._online_train_budget = 0.0

    def _make_value_model(self, feat_dim: int):
        if self.config.use_twohot_value:
            return TwoHotSymlogHead(
                feat_dim=feat_dim,
                hidden_dim=self.config.value_hidden_dim,
                num_layers=self.config.value_num_layers,
                num_bins=self.config.twohot_bins,
                low=self.config.twohot_low,
                high=self.config.twohot_high,
            )
        return DreamerValue(
            DreamerValueConfig(
                feat_dim=feat_dim,
                hidden_dim=self.config.value_hidden_dim,
                num_layers=self.config.value_num_layers,
            )
        )

    def _value_forward(self, feat: torch.Tensor, denormalize: bool = True) -> torch.Tensor:
        return self._head_to_scalar(self.value, feat, denormalize=denormalize)

    def _slow_value_forward(self, feat: torch.Tensor, denormalize: bool = True) -> torch.Tensor:
        if self.slow_value is None:
            return self._value_forward(feat, denormalize=denormalize)
        return self._head_to_scalar(self.slow_value, feat, denormalize=denormalize)

    def _head_to_scalar(self, module, feat: torch.Tensor, denormalize: bool = True) -> torch.Tensor:
        if isinstance(module, TwoHotSymlogHead):
            logits = module(feat)
            value = symexp(twohot_mean(
                logits,
                num_bins=self.config.twohot_bins,
                low=self.config.twohot_low,
                high=self.config.twohot_high,
            ))
        else:
            value = module(feat)
        if denormalize and self.config.use_return_norm:
            return self.return_normalizer.denormalize(value)
        return value

    def _value_loss(self, feat: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.reshape(-1)
        if self.config.use_return_norm:
            target = self.return_normalizer.normalize(target, update=False)
        if isinstance(self.value, TwoHotSymlogHead):
            target = symlog(target) if self.config.use_symlog_reward else target
            logits = self.value(feat)
            return -twohot_logprob(
                logits,
                target,
                num_bins=self.config.twohot_bins,
                low=self.config.twohot_low,
                high=self.config.twohot_high,
            ).mean()
        return F.mse_loss(self.value(feat), target)

    def _sync_slow_value(self):
        if self.slow_value is None:
            return
        rate = float(self.config.slow_value_rate)
        with torch.no_grad():
            for slow_param, fast_param in zip(self.slow_value.parameters(), self.value.parameters()):
                slow_param.data.lerp_(fast_param.data, rate)

    def _replay_value_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            outputs = self.world_model(batch, None)
            feat = outputs["feat"]
        flat_feat = feat.reshape(-1, feat.shape[-1])
        slow_values = self._slow_value_forward(flat_feat).view(feat.shape[0], feat.shape[1])
        current_values = self._value_forward(flat_feat).view(feat.shape[0], feat.shape[1])
        bootstrap = slow_values[:, -1]
        replay_returns = lambda_return(
            rewards=batch["reward"].transpose(0, 1),
            values=torch.cat([slow_values.transpose(0, 1), bootstrap.unsqueeze(0)], dim=0),
            continues=(1.0 - batch["done"]).transpose(0, 1),
            gamma=self.config.gamma,
            lambda_=self.config.lambda_,
        ).transpose(0, 1)
        repval_loss = self._value_loss(flat_feat, replay_returns.detach())
        slow_reg = F.mse_loss(current_values, slow_values.detach())
        return repval_loss, slow_reg

    def reset_latent(self, batch_size: int = 1):
        self.latent_state = self.world_model.rssm.init_state(batch_size, self.device)
        self.latent_context = self.world_model.initial_context(batch_size, self.device)
        self.last_context_metrics = self._default_context_metrics()
        return self.latent_state

    def update_latent(self, obs: np.ndarray, prev_action: np.ndarray, is_first: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.config.observation.mode == "vector":
            obs_t = obs_t.view(1, -1)
        action_t = torch.as_tensor(prev_action, dtype=torch.float32, device=self.device).view(1, -1)
        first_t = torch.as_tensor([float(is_first)], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            embed = self.world_model.encoder(obs_t)
            self.latent_state, _ = self.world_model.rssm.observe_step(
                state=self.latent_state,
                embed=embed,
                action=action_t,
                is_first=first_t,
            )
            feature_details = self.world_model.get_augmented_feat(
                self.latent_state,
                prev_context=self.latent_context,
                is_first=first_t,
                return_details=True,
            )
            self.latent_context = feature_details["context"].detach() if feature_details["context"] is not None else None
            self.last_context_metrics = self._context_metrics_from_details(feature_details)
        return self.latent_state

    def act(self, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            base_feat = self.world_model.get_base_feat(self.latent_state)
            feat = self.world_model.concat_context(base_feat, self.latent_context)
            action = self.actor.mode(feat) if deterministic else self.actor.sample(feat)[0]
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def train_world_model(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.world_model.train()
        outputs = self.world_model(batch, self.loss_config)
        loss = outputs["loss"]
        self.world_model_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.config.grad_clip)
        self.world_model_optimizer.step()
        metrics = {}
        for key, value in outputs["metrics"].items():
            if key == "priority":
                metrics[key] = value.detach().cpu().tolist()
            else:
                metrics[key] = float(value.detach().cpu())
        return metrics

    def _set_world_model_trainable(self, trainable: bool):
        for param in self.world_model.parameters():
            param.requires_grad_(trainable)

    def _posterior_start_state(self, batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
        with torch.no_grad():
            outputs = self.world_model(batch, None)
            post = outputs["post"]
            context = outputs.get("context")
        batch_size, seq_len, _ = post["h"].shape
        if self.config.imag_last and self.config.imag_last > 0:
            steps = min(int(self.config.imag_last), seq_len)
            state = {
                key: value[:, -steps:].reshape(batch_size * steps, *value.shape[2:]).detach()
                for key, value in post.items()
            }
            context_state = (
                context[:, -steps:].reshape(batch_size * steps, context.shape[-1]).detach() if context is not None else None
            )
            return state, context_state
        state = {key: value.reshape(batch_size * seq_len, *value.shape[2:]).detach() for key, value in post.items()}
        context_state = context.reshape(batch_size * seq_len, context.shape[-1]).detach() if context is not None else None
        return state, context_state

    def _discount_weights(self, continues: torch.Tensor) -> torch.Tensor:
        discounts = self.config.gamma * continues.detach()
        prefix = torch.ones(1, discounts.shape[1], dtype=discounts.dtype, device=discounts.device)
        if discounts.shape[0] == 1:
            return prefix
        return torch.cat([prefix, torch.cumprod(discounts[:-1], dim=0)], dim=0)

    def train_actor_value(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.actor.train()
        self.value.train()
        start_state, start_context = self._posterior_start_state(batch)
        self._set_world_model_trainable(False)

        imagined = imagine_rollout(
            world_model=self.world_model,
            actor=self.actor,
            start_state=start_state,
            horizon=self.config.imagination_horizon,
            start_context=start_context,
        )
        feats = imagined["feats"]
        rewards = imagined["rewards"]
        continues = imagined["continues"]
        entropies = imagined["entropies"]
        log_probs = imagined["log_probs"]
        flat_feats = feats.reshape(-1, feats.shape[-1])
        values = self._value_forward(flat_feats).view(feats.shape[0], feats.shape[1])
        last_state = {key: value[-1] for key, value in imagined["states"].items()}
        bootstrap_feat = self.world_model.concat_context(
            self.world_model.get_base_feat(last_state),
            imagined.get("last_context"),
        )
        bootstrap = self._slow_value_forward(bootstrap_feat).unsqueeze(0)
        all_values = torch.cat([values, bootstrap], dim=0)
        returns = lambda_return(
            rewards=rewards,
            values=all_values,
            continues=continues,
            gamma=self.config.gamma,
            lambda_=self.config.lambda_,
        )
        if self.config.use_return_norm:
            self.return_normalizer.update(returns.detach())
        weights = self._discount_weights(continues)
        advantage = returns.detach() - values.detach()
        if self.config.use_advantage_norm:
            norm_advantage = self.advantage_normalizer.normalize(advantage, update=True)
        else:
            norm_advantage = advantage
        actor_loss = -(weights * (log_probs * norm_advantage + self.config.entropy_coef * entropies)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.actor_grad_clip)
        self.actor_optimizer.step()

        value_loss = self._value_loss(flat_feats.detach(), returns.detach())
        repval_loss = torch.zeros((), device=self.device)
        slowreg_loss = torch.zeros((), device=self.device)
        if self.config.repval_loss:
            repval_loss, slowreg_loss = self._replay_value_loss(batch)
            value_loss = value_loss + self.config.repval_scale * repval_loss + self.config.slow_value_rate * slowreg_loss
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.value_grad_clip)
        self.value_optimizer.step()
        self._sync_slow_value()
        self._set_world_model_trainable(True)

        value_pred = self._value_forward(flat_feats.detach()).view(feats.shape[0], feats.shape[1])
        return {
            "actor_loss": float(actor_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "repval_loss": float(repval_loss.detach().cpu()),
            "slowreg_loss": float(slowreg_loss.detach().cpu()),
            "imagined_return": float(returns.detach().mean().cpu()),
            "value_pred": float(value_pred.detach().mean().cpu()),
            "return_norm_mean": float(self.return_normalizer.mean),
            "return_norm_std": float(self.return_normalizer.std),
            "adv_norm_mean": float(self.advantage_normalizer.mean),
            "adv_norm_std": float(self.advantage_normalizer.std),
            "entropy": float(entropies.detach().mean().cpu()),
        }

    def update_from_replay(
        self,
        replay: "EpisodeReplayBuffer",
        include_contact: bool = False,
        include_grasp: bool = False,
    ) -> tuple[dict[str, float], dict[str, object]]:
        metadata: dict[str, object] = {
            "replay_size": len(replay),
            "warmup_steps": self._online_warmup_steps,
            "train_calls": self._online_train_calls + 1,
            "device": str(self.device),
        }

        self._online_train_calls += 1
        if len(replay) < self._online_warmup_steps or self._online_train_calls % self._online_train_every != 0:
            return {}, metadata

        self._online_train_budget += self._online_train_ratio
        budget_updates = int(self._online_train_budget)
        if budget_updates <= 0:
            return {}, metadata

        num_updates = budget_updates
        if self._online_max_updates_per_step > 0:
            num_updates = min(num_updates, self._online_max_updates_per_step)
        self._online_train_budget -= num_updates

        if not replay.can_sample(self._online_batch_size, self._online_seq_len, include_contact, include_grasp):
            if include_contact or include_grasp:
                include_contact = False
                include_grasp = False
            if not replay.can_sample(self._online_batch_size, self._online_seq_len, include_contact, include_grasp):
                return {}, metadata

        aggregates: dict[str, float] = {}
        for _ in range(num_updates):
            batch = replay.sample_batch(
                batch_size=self._online_batch_size,
                seq_len=self._online_seq_len,
                device=self.device,
                include_contact=include_contact,
                include_grasp=include_grasp,
            )
            sample_refs = batch.pop("_sample_refs", [])
            wm_metrics = self.train_world_model(batch)
            if "priority" in wm_metrics:
                replay.update_priorities(sample_refs, wm_metrics["priority"])
                del wm_metrics["priority"]
            actor_metrics = self.train_actor_value(batch)
            for metrics in (wm_metrics, actor_metrics):
                for key, value in metrics.items():
                    aggregates[key] = aggregates.get(key, 0.0) + float(value)

        metrics = {key: value / num_updates for key, value in aggregates.items()}
        metrics["num_updates"] = float(num_updates)
        metrics["train_budget"] = float(self._online_train_budget)
        metadata.update(
            {
                "batch_size": self._online_batch_size,
                "seq_len": self._online_seq_len,
                "include_contact": include_contact,
                "include_grasp": include_grasp,
            }
        )
        return metrics, metadata

    def state_dict(self) -> dict:
        return {
            "config": self.config.asdict(),
            "loss_config": asdict(self.loss_config),
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "value": self.value.state_dict(),
            "slow_value": self.slow_value.state_dict() if self.slow_value is not None else None,
            "return_normalizer": self.return_normalizer.state_dict(),
            "advantage_normalizer": self.advantage_normalizer.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "online_update_state": {
                "train_calls": self._online_train_calls,
                "train_budget": self._online_train_budget,
            },
        }

    def _load_module_partial(self, module, state_dict: dict, module_name: str) -> dict[str, object]:
        current = module.state_dict()
        compatible = {}
        unexpected = []
        shape_mismatches = []
        for key, value in state_dict.items():
            if key not in current:
                unexpected.append(key)
                continue
            if current[key].shape != value.shape:
                shape_mismatches.append(
                    {
                        "key": key,
                        "expected": tuple(current[key].shape),
                        "got": tuple(value.shape),
                    }
                )
                continue
            compatible[key] = value
        load_result = module.load_state_dict(compatible, strict=False)
        return {
            "module": module_name,
            "missing": list(load_result.missing_keys),
            "unexpected": unexpected,
            "shape_mismatches": shape_mismatches,
        }

    def _load_payload(self, payload: dict, allow_partial: bool = False):
        reports = []
        modules = [
            ("world_model", self.world_model, payload["world_model"]),
            ("actor", self.actor, payload["actor"]),
            ("value", self.value, payload["value"]),
        ]
        if self.slow_value is not None and payload.get("slow_value") is not None:
            modules.append(("slow_value", self.slow_value, payload["slow_value"]))
        for module_name, module, state_dict in modules:
            if not allow_partial:
                module.load_state_dict(state_dict)
                continue
            reports.append(self._load_module_partial(module, state_dict, module_name))
        if "return_normalizer" in payload:
            self.return_normalizer = RunningNormalizer.from_state_dict(payload["return_normalizer"])
        if "advantage_normalizer" in payload:
            self.advantage_normalizer = RunningNormalizer.from_state_dict(payload["advantage_normalizer"])
        if "world_model_optimizer" in payload:
            try:
                self.world_model_optimizer.load_state_dict(payload["world_model_optimizer"])
            except (ValueError, RuntimeError):
                reports.append({"module": "world_model_optimizer", "warning": "optimizer state skipped"})
        if "actor_optimizer" in payload:
            try:
                self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            except (ValueError, RuntimeError):
                reports.append({"module": "actor_optimizer", "warning": "optimizer state skipped"})
        if "value_optimizer" in payload:
            try:
                self.value_optimizer.load_state_dict(payload["value_optimizer"])
            except (ValueError, RuntimeError):
                reports.append({"module": "value_optimizer", "warning": "optimizer state skipped"})
        if "online_update_state" in payload:
            state = payload["online_update_state"]
            self._online_train_calls = int(state.get("train_calls", 0))
            self._online_train_budget = float(state.get("train_budget", 0.0))
        partial = any(
            report.get("missing") or report.get("unexpected") or report.get("shape_mismatches") or report.get("warning")
            for report in reports
        )
        if partial:
            print(
                "[DreamerV3] Loaded checkpoint with partial parameter reuse. "
                "This is expected when enabling thick_context on an older checkpoint."
            )
        self._last_load_report = {"partial": partial, "reports": reports}
        return self._last_load_report

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def from_state_dict(
        cls,
        payload: dict,
        device: str | None = None,
        model_config: DreamerV3ModelConfig | None = None,
    ):
        config_override = model_config is not None
        if model_config is None:
            aux = DreamerAuxConfig(**payload["config"]["aux"])
            observation = DreamerObservationSpec(**payload["config"]["observation"])
            config_data = dict(payload["config"])
            config_data["aux"] = aux
            config_data["observation"] = observation
            if device is not None:
                config_data["device"] = device
            model_config = DreamerV3ModelConfig(**config_data)
        loss_config = None if config_override else WorldModelLossConfig(**payload["loss_config"])
        agent = cls(model_config, loss_config)
        agent._load_payload(payload, allow_partial=bool(model_config.thick_context.enabled))
        return agent

    @classmethod
    def load(cls, path: str | Path, device: str | None = None):
        return cls.from_state_dict(torch.load(path, map_location=device or "cpu"), device=device)
