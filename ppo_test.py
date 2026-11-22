"""加载训练好的 PPO 模型，运行若干回合并通过 3D 动画渲染查看着陆效果。

运行方式（在项目根目录下）：
    conda run -n wta_sim python ppo_test.py
"""

from pathlib import Path

import numpy as np

# 在导入 rocket_gym（其内部会 import pyplot）之前，尽量切换到可交互后端，
# 避免默认 Agg 后端导致只出图不弹窗。
import matplotlib

_current = matplotlib.get_backend().lower()
if not _current.startswith(("tk", "qt", "wx", "gtk")):
    for _b in ("TkAgg", "Qt5Agg", "QtAgg"):
        try:
            matplotlib.use(_b)
            break
        except Exception:
            continue

import gymnasium as gym
from gymnasium import register
from stable_baselines3 import PPO

BASE_DIR = Path(__file__).resolve().parent
ENV_ID = "rocket_environment-v0"
MODEL_PATH = BASE_DIR / "model_ppo"  # 想看历史版本可改成 model_ppo_v2 / model_ppo_v1

NUM_EPISODES = 3
SEED_OFFSET = 100
RENDER = True
RENDER_PAUSE = 0.3  # 每帧暂停秒数，越小动画越快


def main():
    register(id=ENV_ID, entry_point="rocket_gym:Rocket_Model_6DoF")
    env = gym.make(ENV_ID)
    model = PPO.load(str(MODEL_PATH), device="cpu")

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset(seed=SEED_OFFSET + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1

        x = env.unwrapped.x
        end_speed = float(np.linalg.norm(x[4:7]))
        pos_err = float(np.linalg.norm(x[1:4]))
        reasons = info.get("termination_reasons")
        print(
            f"Episode {ep}: ret={ep_ret:7.3f}  len={ep_len:3d}  "
            f"end_speed={end_speed:5.3f}  pos_err={pos_err:5.3f}  "
            f"reasons={reasons}"
        )

        if RENDER:
            env.unwrapped.render(pause=RENDER_PAUSE)

    env.close()


if __name__ == "__main__":
    main()
