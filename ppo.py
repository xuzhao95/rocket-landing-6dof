from pathlib import Path

import gymnasium as gym
from gymnasium import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

BASE_DIR = Path(__file__).resolve().parent
ENV_ID = "rocket_environment-v0"
N_ENVS = 8
TOTAL_TIMESTEPS = 25_000_000
SEED = 42
CHECKPOINT_DIR = BASE_DIR / "checkpoints"


def main():
    register(
        id=ENV_ID,
        entry_point="rocket_gym:Rocket_Model_6DoF",
    )
    env = make_vec_env(ENV_ID, n_envs=N_ENVS, seed=SEED)
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="rocket_ppo",
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        ),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        verbose=1,
        device="cpu",
        seed=SEED,
    )

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
        )
        model.save(BASE_DIR / "model_ppo")
    finally:
        env.close()

    test_env = gym.make(ENV_ID)
    try:
        obs, _ = test_env.reset(seed=SEED)
        terminated = truncated = False
        episode_return = 0.0
        episode_length = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = test_env.step(action)
            episode_return += reward
            episode_length += 1

        print(
            f"TestEpRet={episode_return:.3f}, "
            f"TestEpLen={episode_length}"
        )
    finally:
        test_env.close()


if __name__ == "__main__":
    main()
