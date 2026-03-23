from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from communication.tcp_client import SimulationClient
from environments.point_tracking.point_tracking_env import PointTrackingEnv


def make_env(log_save=False):
    simulation = SimulationClient(host='127.0.0.1', port=8888, steps=6, env_name="control", log_save=log_save)
    environment = PointTrackingEnv(simulation_client=simulation, max_steps=200, random_init=True)
    return environment


if __name__ == "__main__":
    train = True
    if train:
        env = DummyVecEnv([make_env])
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
            device='cpu',
            tensorboard_log="./ppo_tracking_tensorboard/"
        )
        for i in range(10):
            # 训练
            model.learn(total_timesteps=100)
            model.save("ppo_point_tracking")
    else:
        env = make_env(log_save=True)
        model = PPO.load("ppo_point_tracking", env=env, device='cpu')
        # 测试
        obs, info = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
    env.close()
