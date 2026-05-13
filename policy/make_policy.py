from policy.mlp import MLPPolicyFolder
from policy.random import RandomPolicyFolder
from policy.dreamerv3 import DreamerV3PolicyFolder


def make_policy(config, env, exam_dir=None):
    policy_type = config["type"]

    if policy_type == "random":
        random_config = config.get("random", {})
        return RandomPolicyFolder(
            action_dim=env.action_dim,
            action_low=random_config.get("action_low", -1.0),
            action_high=random_config.get("action_high", 1.0),
        )

    if policy_type == "mlp":
        # TODO: 把 hidden_size / noise_std 等超参数从 YAML 暴露出来；现在工厂函数把策略配置基本丢掉了。
        return MLPPolicyFolder(action_dim=env.action_dim, obs_dim=env.obs_dim)

    if policy_type == "dreamerv3":
        observation_example = env.reset()
        return DreamerV3PolicyFolder(
            action_dim=env.action_dim,
            observation_example=observation_example,
            policy_config=config,
            exam_dir=exam_dir,
        )

    raise ValueError(f"Unsupported policy type: {policy_type}")
