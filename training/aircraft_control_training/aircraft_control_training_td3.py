from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from communication.tcp_client import SimulationClient
from environments.aircraft_control.aircraft_control_env import AircraftControlEnv
import numpy as np
import os

# 创建 CheckpointCallback
# save_freq=10000 表示每 10000 步保存一次模型
checkpoint_callback = CheckpointCallback(
    save_freq=100000,                      # 保存间隔（步数）
    save_path="./models/",                # 保存路径
    name_prefix="td3_control",              # 文件名前缀
    save_replay_buffer=True,              # 是否同时保存经验回放缓冲区
    save_vecnormalize=True                # 如果使用了 VecNormalize，一起保存
)


def make_env(log_save=False):
    simulation = SimulationClient(host='127.0.0.1', port=8888, steps=6, env_name="control", log_save=log_save)
    environment = AircraftControlEnv(simulation_client=simulation, max_steps=400, random_init=True)
    return environment


if __name__ == "__main__":
    train = True
    continue_training = True
    if train:
        env = DummyVecEnv([make_env])
        # 获取动作空间的维度
        n_actions = env.action_space.shape[0]

        # 创建动作噪声（均值为0，标准差为0.1 * 动作范围）
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions)
        )

        # 创建带噪声的TD3模型
        model = TD3(
            "MlpPolicy",
            env,
            action_noise=action_noise,  # 添加动作噪声
            verbose=1,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            tensorboard_log="./td3_control_tensorboard/"
        )
        if continue_training:
            replay_buffer_path = "./models/td3_control_replay_buffer_4000000_steps.pkl"
            # 加载回放缓冲区
            if os.path.exists(replay_buffer_path):
                model.load_replay_buffer(path=replay_buffer_path)
                print(f"成功加载回放缓冲区，包含 {model.replay_buffer.size()} 个样本")

        # 训练时传入回调函数
        model.learn(
            total_timesteps=10000000,
            callback=checkpoint_callback  # 关键：将回调传入 learn 方法
        )
    else:
        env = make_env(log_save=True)
        model = TD3.load("./models/td3_control_3900000_steps.zip", env=env)
        # replay_buffer_path = "./models/td3_control_replay_buffer_4000000_steps.pkl"
        # # 加载回放缓冲区
        # if os.path.exists(replay_buffer_path):
        #     model.load_replay_buffer(path=replay_buffer_path)
        #     print(f"成功加载回放缓冲区，包含 {model.replay_buffer.size()} 个样本")

        # 测试
        obs, info = env.reset()
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        env.close()
