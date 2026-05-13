from pathlib import Path

import imageio.v2 as imageio


def save_frame(env, out_dir, step):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"frame_step_{step:06d}.png"
    # TODO: 对 env.render() 的返回做 shape/dtype 校验；不同后端可能返回 None、float 图像或通道顺序不一致的数据。
    imageio.imwrite(path, env.render())
    return path


def save_gif(frames, path, fps=20):
    if not frames:
        raise ValueError("Cannot save GIF without frames")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, format="GIF", fps=fps)
    return path
