from policy.registry import load_policy_class


def make_policy(config, env, exam_dir=None):
    policy_type = str(config["type"]).split("@", 1)[0]
    policy_cls = load_policy_class(config)

    if policy_type == "random":
        random_config = config.get("random", {})
        return policy_cls(
            action_dim=env.action_dim,
            action_low=random_config.get("action_low", -1.0),
            action_high=random_config.get("action_high", 1.0),
        )

    if policy_type == "mlp":
        # TODO: 把 hidden_size / noise_std 等超参数从 YAML 暴露出来；现在工厂函数把策略配置基本丢掉了。
        return policy_cls(action_dim=env.action_dim, obs_dim=env.obs_dim)

    if policy_type in {
        "dreamerv3",
        "EGO",
        "DreamerV3milestoneV2",
        "dreamerv3_milestoneV2",
        "dreamerv3_milestoneV3",
        "dreamerv3milestoneV2",
        "DreamerV3milestoneV3",
        "dreamerv3milestoneV1",
        "better_dreamerv3milestoneV1",
        "better_dreamerv3_milestoneV2",
        "better_dreamerv3_milestoneV3",
        "better_dreamerv3_by_miscanthus",
        "better_dreamerv3milestoneV4",
        "better_dreamerv3milestoneV5",
        "better_dreamerv3milestoneV6",
        "better_dreamerv3milestoneV7",
        "better_dreamerv3milestoneV8",
    }:
        observation_example = env.reset()
        return policy_cls(
            action_dim=env.action_dim,
            observation_example=observation_example,
            policy_config=config,
            exam_dir=exam_dir,
        )

    if policy_type == "operator_token_probe":
        observation_example = env.reset()
        return policy_cls(
            action_dim=env.action_dim,
            observation_example=observation_example,
            policy_config=config,
            exam_dir=exam_dir,
        )

    raise ValueError(f"Unsupported policy type: {policy_type}")
