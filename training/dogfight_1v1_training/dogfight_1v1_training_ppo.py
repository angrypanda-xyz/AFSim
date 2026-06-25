from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from communication.tcp_client import SimulationClient
from environments.dogfight_1v1.dogfight_1v1_env import DogFight1v1Env
from functools import partial


def make_env(steps, max_steps, log_save=False):
    simulation = SimulationClient(host='127.0.0.1', port=8888, steps=steps, environment="dogfight", log_save=log_save)
    environment = DogFight1v1Env(simulation_client=simulation, max_steps=max_steps, random_init=True)
    return Monitor(environment)


if __name__ == "__main__":
    train = True
    if train:
        env_fns = [partial(make_env, steps=20, max_steps=1000, log_save=False) for _ in range(5)]
        env = SubprocVecEnv(env_fns)
        # env = DummyVecEnv([make_env])
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            device="cpu",
            tensorboard_log="./dogfight_1v1_tensorboard/"
        )
        for i in range(10):
            # 训练
            model.learn(total_timesteps=10)
            model.save("ppo_dogfight_1v1")
    else:
        # 测试
        env = make_env(log_save=True)
        model = PPO.load("ppo_dogfight_1v1", env=env, device="cpu")
        obs, info = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    env.close()
