from .registry import load_env_components


def _maybe_wrap_persistent(env, config):
    components = load_env_components(config)
    PersistentExplorationConfig = components["PersistentExplorationConfig"]
    PersistentExplorationEnv = components["PersistentExplorationEnv"]
    persistent_cfg = dict(config.get("persistent_exploration", {}))
    if not bool(persistent_cfg.get("enabled", False)):
        return env
    return PersistentExplorationEnv(env, PersistentExplorationConfig(**persistent_cfg))


def make_env(config):
    components = load_env_components(config)
    DMControlEnv = components["DMControlEnv"]
    BallInCupTakeOutEnv = components["BallInCupTakeOutEnv"]
    MetaWorldEnv = components["MetaWorldEnv"]
    if config["type"] == "dmcontrol":
        render_config = config.get("render", {})
        observation_config = config.get("observation", {})
        camera_indices = observation_config.get("camera_indices")
        if camera_indices is None:
            camera_indices = render_config.get("camera_indices")
        domain_name = config["domain_name"]
        task_name = config["task_name"]
        if domain_name == "ball_in_cup" and task_name in {"release", "take_out"}:
            env = BallInCupTakeOutEnv(
                render_height=render_config.get("height", 240),
                render_width=render_config.get("width", 320),
                camera_id=render_config.get("camera_id", 0),
                camera_indices=camera_indices,
                observation_type=observation_config.get("type", "vector"),
                num_cams=observation_config.get("num_cams", 1),
                render_backend=render_config.get("backend"),
                render_backend_priority=render_config.get("backend_priority"),
                allow_software_render_fallback=bool(render_config.get("allow_software_render_fallback", False)),
                render_enabled=bool(render_config.get("enabled", False)),
            )
            return _maybe_wrap_persistent(env, config)
        env = DMControlEnv(
            domain_name=domain_name,
            task_name=task_name,
            render_height=render_config.get("height", 240),
            render_width=render_config.get("width", 320),
            camera_id=render_config.get("camera_id", 0),
            camera_indices=camera_indices,
            observation_type=observation_config.get("type", "vector"),
            num_cams=observation_config.get("num_cams", 1),
            render_backend=render_config.get("backend"),
            render_backend_priority=render_config.get("backend_priority"),
            allow_software_render_fallback=bool(render_config.get("allow_software_render_fallback", False)),
            render_enabled=bool(render_config.get("enabled", False)),
        )
        return _maybe_wrap_persistent(env, config)

    if config["type"] == "peg_insert_side_reverse_sparse":
        PegInsertSideReverseSparseStandaloneEnv = components["PegInsertSideReverseSparseStandaloneEnv"]
        render_config = config.get("render", {})
        observation_config = config.get("observation", {})
        camera_indices = observation_config.get("camera_indices")
        if camera_indices is None:
            camera_indices = render_config.get("camera_indices")
        metaworld_config = config.get("metaworld", {})
        env = PegInsertSideReverseSparseStandaloneEnv(
            render_height=render_config.get("height", 240),
            render_width=render_config.get("width", 320),
            camera_id=render_config.get("camera_id", 0),
            camera_indices=camera_indices,
            observation_type=observation_config.get("type", "vector"),
            num_cams=observation_config.get("num_cams", 1),
            render_backend=render_config.get("backend"),
            render_backend_priority=render_config.get("backend_priority"),
            allow_software_render_fallback=bool(render_config.get("allow_software_render_fallback", False)),
            render_enabled=bool(render_config.get("enabled", False)),
            seed=int(metaworld_config.get("seed", 0)),
            reward_function_version=str(metaworld_config.get("reward_function_version", "v2")),
        )
        return _maybe_wrap_persistent(env, config)

    if config["type"] == "metaworld":
        render_config = config.get("render", {})
        observation_config = config.get("observation", {})
        camera_indices = observation_config.get("camera_indices")
        if camera_indices is None:
            camera_indices = render_config.get("camera_indices")
        metaworld_config = config.get("metaworld", {})
        env = MetaWorldEnv(
            env_name=config["task_name"],
            render_height=render_config.get("height", 240),
            render_width=render_config.get("width", 320),
            camera_id=render_config.get("camera_id", 0),
            camera_indices=camera_indices,
            observation_type=observation_config.get("type", "vector"),
            num_cams=observation_config.get("num_cams", 1),
            render_backend=render_config.get("backend"),
            render_backend_priority=render_config.get("backend_priority"),
            allow_software_render_fallback=bool(render_config.get("allow_software_render_fallback", False)),
            render_enabled=bool(render_config.get("enabled", False)),
            seed=int(metaworld_config.get("seed", 0)),
            reward_function_version=str(metaworld_config.get("reward_function_version", "v2")),
        )
        return _maybe_wrap_persistent(env, config)

    # TODO: 文档和代码里已经出现 IsaacEnv，但工厂函数还不支持 isaac 配置分支，接口处于半实现状态。
    raise ValueError(f"Unsupported env type: {config['type']}")
