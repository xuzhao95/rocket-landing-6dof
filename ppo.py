from pathlib import Path

import gymnasium as gym
from gymnasium import register
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env

BASE_DIR = Path(__file__).resolve().parent
ENV_ID = "rocket_environment-v0"
N_ENVS = 8
TOTAL_TIMESTEPS = 5_000_000
SEED = 42
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
INITIAL_MODEL = BASE_DIR / "model_ppo.zip"


def _latest_checkpoint() -> Path | None:
    candidates = sorted(
        CHECKPOINT_DIR.glob("rocket_ppo_*_steps.zip"),
        key=lambda p: int(p.stem.split("_")[-2]),
    )
    return candidates[-1] if candidates else None


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

    resume_from = _latest_checkpoint()
    if resume_from is not None:
        model = PPO.load(resume_from, env=env, device="cpu")
        remaining = max(0, TOTAL_TIMESTEPS - model.num_timesteps)
        reset_num_timesteps = False
        print(
            f"Resuming from {resume_from.name} "
            f"({model.num_timesteps} steps done, {remaining} remaining)"
        )
    elif INITIAL_MODEL.exists():
        model = PPO.load(INITIAL_MODEL, env=env, device="cpu")
        remaining = TOTAL_TIMESTEPS
        reset_num_timesteps = True
        print(
            f"Fine-tuning from {INITIAL_MODEL.name} "
            f"for {remaining} steps"
        )
    else:
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
        remaining = TOTAL_TIMESTEPS
        reset_num_timesteps = True
        print(f"Starting fresh training for {remaining} steps")

    try:
        model.learn(
            total_timesteps=remaining,
            callback=checkpoint_callback,
            reset_num_timesteps=reset_num_timesteps,
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
