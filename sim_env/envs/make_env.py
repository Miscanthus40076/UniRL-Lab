from .dm_control_env import DMControlEnv


def make_env(config):
    if config["type"] == "dmcontrol":
        render_config = config.get("render", {})
        observation_config = config.get("observation", {})
        return DMControlEnv(
            domain_name=config["domain_name"],
            task_name=config["task_name"],
            render_height=render_config.get("height", 240),
            render_width=render_config.get("width", 320),
            camera_id=render_config.get("camera_id", 0),
            observation_type=observation_config.get("type", "vector"),
            num_cams=observation_config.get("num_cams", 1),
            render_backend=render_config.get("backend"),
            render_backend_priority=render_config.get("backend_priority"),
            allow_software_render_fallback=bool(render_config.get("allow_software_render_fallback", False)),
            render_enabled=bool(render_config.get("enabled", False)),
        )

    # TODO: 文档和代码里已经出现 IsaacEnv，但工厂函数还不支持 isaac 配置分支，接口处于半实现状态。
    raise ValueError(f"Unsupported env type: {config['type']}")
