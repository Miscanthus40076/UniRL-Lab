from __future__ import annotations

from copy import deepcopy
from functools import partial as bind
from pathlib import Path
import sys

import yaml

from sim_env.envs.make_env import make_env
from sim_env.wrappers.registry import apply_wrappers, wrapper_config_from_exam

from .base import Trainer


REF_DREAMERV3_ROOT = Path(__file__).resolve().parents[2] / "policy" / "REF_policy" / "dreamerv3"


class RefDreamerV3Trainer(Trainer):
    trainer_name = "ref_dreamerv3"

    def run(self) -> dict:
        output_dir = self.ensure_output_dir()
        self._ensure_ref_import_path()
        elements, embodied, dreamer_agent = self._import_ref_stack()

        config = self._build_ref_config(elements, output_dir)
        self._write_run_manifest(output_dir, config)
        args = self._build_run_args(elements, config)

        embodied.run.train(
            bind(self._make_agent, elements, dreamer_agent, config),
            bind(self._make_replay, elements, embodied, config, "replay"),
            bind(self._make_env, config),
            bind(self._make_stream, embodied, config),
            bind(self._make_logger, elements, config),
            args,
        )

        return {
            "trainer": self.trainer_name,
            "output_dir": output_dir,
            "manifest_path": output_dir / "run_manifest.json",
        }

    def _ensure_ref_import_path(self):
        ref_root = str(REF_DREAMERV3_ROOT)
        if ref_root not in sys.path:
            sys.path.insert(0, ref_root)

    def _import_ref_stack(self):
        try:
            import elements
            import embodied
            from dreamerv3.agent import Agent
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ref_dreamerv3 requires the official DreamerV3 dependencies. "
                "Install policy/REF_policy/dreamerv3/requirements.txt in the active environment."
            ) from exc
        return elements, embodied, Agent

    def _ref_policy_config(self) -> dict:
        policy = dict(self.config.get("policy", {}))
        return dict(policy.get("ref_dreamerv3", {}))

    def _build_ref_config(self, elements, output_dir: Path):
        configs_path = REF_DREAMERV3_ROOT / "dreamerv3" / "configs.yaml"
        with configs_path.open("r", encoding="utf-8") as handle:
            configs = yaml.safe_load(handle)

        ref_cfg = self._ref_policy_config()
        selected = ref_cfg.get("configs", ["defaults"])
        if isinstance(selected, str):
            selected = [selected]
        if not selected:
            selected = ["defaults"]

        config = elements.Config(configs["defaults"])
        for name in selected:
            if name == "defaults":
                continue
            if name not in configs:
                supported = ", ".join(sorted(configs))
                raise ValueError(f"Unsupported ref_dreamerv3 config preset: {name}. Supported: {supported}")
            config = config.update(configs[name])

        train = dict(self.config.get("train", {}))
        overrides = deepcopy(ref_cfg.get("overrides", {}))
        overrides = self._deep_merge(
            {
                "logdir": str(output_dir),
                "seed": int(train.get("seed", self.seed)),
                "script": "train",
                "run": {
                    "steps": int(train.get("total_steps", overrides.get("run", {}).get("steps", 1000))),
                },
            },
            overrides,
        )
        config = config.update(overrides)
        return config

    def _build_run_args(self, elements, config):
        return elements.Config(
            **config.run,
            replica=config.replica,
            replicas=config.replicas,
            logdir=config.logdir,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            report_length=config.report_length,
            consec_train=config.consec_train,
            consec_report=config.consec_report,
            replay_context=config.replay_context,
        )

    def _make_agent(self, elements, Agent, config):
        env = self._make_env(config, 0)
        notlog = lambda key: not key.startswith("log/")
        obs_space = {key: value for key, value in env.obs_space.items() if notlog(key)}
        act_space = {key: value for key, value in env.act_space.items() if key != "reset"}
        env.close()
        if config.random_agent:
            import embodied

            return embodied.RandomAgent(obs_space, act_space)
        return Agent(
            obs_space,
            act_space,
            elements.Config(
                **config.agent,
                logdir=config.logdir,
                seed=config.seed,
                jax=config.jax,
                batch_size=config.batch_size,
                batch_length=config.batch_length,
                replay_context=config.replay_context,
                report_length=config.report_length,
                replica=config.replica,
                replicas=config.replicas,
            ),
        )

    def _make_replay(self, elements, embodied, config, folder, mode="train"):
        batlen = config.batch_length if mode == "train" else config.report_length
        consec = config.consec_train if mode == "train" else config.consec_report
        capacity = config.replay.size if mode == "train" else config.replay.size / 10
        length = consec * batlen + config.replay_context
        if config.batch_size * length > capacity:
            raise ValueError("DreamerV3 replay capacity is smaller than batch_size * sequence_length")

        directory = elements.Path(config.logdir) / folder
        if config.replicas > 1:
            directory /= f"{config.replica:05}"
        return embodied.replay.Replay(
            length=length,
            capacity=int(capacity),
            online=config.replay.online,
            chunksize=config.replay.chunksize,
            directory=directory,
        )

    def _make_env(self, config, index: int, **overrides):
        env_cfg = deepcopy(self.config.get("env", {}))
        wrapper_cfg = wrapper_config_from_exam(self.config)
        if not wrapper_cfg:
            wrapper_cfg = [{"type": "simer_to_embodied", "version": "v1"}]
        train = dict(self.config.get("train", {}))
        for entry in wrapper_cfg:
            if entry.get("type") == "simer_to_embodied" and "max_episode_steps" not in entry:
                if train.get("max_episode_steps") is not None:
                    entry["max_episode_steps"] = int(train["max_episode_steps"])
        env = make_env(env_cfg)
        env = apply_wrappers(env, wrapper_cfg)
        return self._wrap_ref_env(env, config)

    def _wrap_ref_env(self, env, config):
        import embodied

        for name, space in env.act_space.items():
            if not space.discrete:
                env = embodied.wrappers.NormalizeAction(env, name)
        env = embodied.wrappers.UnifyDtypes(env)
        env = embodied.wrappers.CheckSpaces(env)
        for name, space in env.act_space.items():
            if not space.discrete:
                env = embodied.wrappers.ClipAction(env, name)
        return env

    def _make_stream(self, embodied, config, replay, mode):
        stream = embodied.streams.Stateless(bind(replay.sample, config.batch_size, mode))
        return embodied.streams.Consec(
            stream,
            length=config.batch_length if mode == "train" else config.report_length,
            consec=config.consec_train if mode == "train" else config.consec_report,
            prefix=config.replay_context,
            strict=(mode == "train"),
            contiguous=True,
        )

    def _make_logger(self, elements, config):
        step = elements.Counter()
        outputs = [
            elements.logger.TerminalOutput(config.logger.filter, "Agent"),
            elements.logger.JSONLOutput(config.logdir, "metrics.jsonl"),
            elements.logger.JSONLOutput(config.logdir, "scores.jsonl", "episode/score"),
        ]
        return elements.Logger(step, outputs, 1)

    def _write_run_manifest(self, output_dir: Path, ref_config) -> Path:
        manifest = {
            "trainer": self.trainer_name,
            "env": dict(self.config.get("env", {})),
            "wrappers": wrapper_config_from_exam(self.config) or [{"type": "simer_to_embodied", "version": "v1"}],
            "policy": dict(self.config.get("policy", {})),
            "artifacts": {
                "metrics_jsonl": str(output_dir / "metrics.jsonl"),
                "scores_jsonl": str(output_dir / "scores.jsonl"),
                "checkpoint_dir": str(output_dir / "ckpt"),
            },
            "ref_dreamerv3": {
                "root": str(REF_DREAMERV3_ROOT),
                "logdir": str(ref_config.logdir),
            },
        }
        return self.write_json(output_dir / "run_manifest.json", manifest)

    def _deep_merge(self, base, override):
        if not isinstance(base, dict) or not isinstance(override, dict):
            return override
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
