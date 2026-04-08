from pyexpat import features
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from torch.cuda import device
from communication.tcp_client import SimulationClient
from environments.point_tracking.point_tracking_env import PointTrackingEnv
from typing import Optional
import datetime
import os
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SaveModelCallback(BaseCallback):
    """每1000步保存一次模型，覆盖之前的模型"""

    def __init__(self, save_path: str, save_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, "ppo_point_tracking")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"已保存模型 (step {self.n_calls}): {model_path}")
        return True


def make_env(steps, max_steps, log_save=False, tack_view=False, render_mode=None):
    def _init():
        simulation = SimulationClient(
            env_num=1,
            host='127.0.0.1',
            port=8888,
            env_name="control",
            steps=steps,
            log_save=log_save,
            tac_view=tack_view,
        )
        simulation.connection()

        env = PointTrackingEnv(
            simulation_client=simulation,
            env_id=0,  # 单环境内部永远是0
            max_steps=max_steps,
            render_mode=render_mode
        )
        return Monitor(env)

    return _init


class LayerNormExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        input_dim = observation_space.shape[0]

        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

    def forward(self, obs):
        return self.net(obs)


def load_model(model_dir: str = None, env=None):
    """
    加载模型

    Args:
        model_dir: 模型目录路径，如果为None则加载最新的模型
        env: 环境实例

    Returns:
        加载的模型
    """
    if model_dir is None:
        model_base_dir = "./ppo_models/"
        if not os.path.exists(model_base_dir):
            raise ValueError(f"模型目录不存在: {model_base_dir}")

        dirs = [d for d in os.listdir(model_base_dir)
                if os.path.isdir(os.path.join(model_base_dir, d))]
        if not dirs:
            raise ValueError("没有找到任何训练模型")

        latest_dir = sorted(dirs)[-1]
        model_dir = os.path.join(model_base_dir, latest_dir)
        print(f"加载最新模型: {model_dir}")

    model_path = os.path.join(model_dir, "ppo_point_tracking")
    return PPO.load(model_path, env=env)


def train():
    # 创建带时间戳的目录
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_save_dir = f"./ppo_models/{timestamp}/"
    tensorboard_log_dir = f"./ppo_tracking_tensorboard/{timestamp}/"

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)

    # 创建向量化环境
    env_num = 16
    max_steps = 1000
    steps = 20
    env = SubprocVecEnv([make_env(steps=steps, max_steps=max_steps) for _ in range(env_num)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    # 创建模型
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        features_extractor_class=LayerNormExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512]))
    model = PPO(
        "MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        target_kl=0.05,
        normalize_advantage=True,
        tensorboard_log=tensorboard_log_dir,
    )

    # 创建自定义回调
    save_callback = SaveModelCallback(save_path=model_save_dir, save_freq=10000, verbose=1)

    # 训练
    model.learn(total_timesteps=int(1e9), callback=save_callback)

    # 训练结束后保存最终模型（覆盖之前的）
    model.save(f"{model_save_dir}/ppo_point_tracking")
    env.save(os.path.join(model_save_dir, "vecnormalize.pkl"))
    env.close()


def test():
    # 测试
    model_base_dir = "./ppo_models/"
    latest_dir = sorted(os.listdir(model_base_dir))[-1]
    # latest_dir = "固定点_1e7"
    model_dir = os.path.join(model_base_dir, latest_dir)

    max_steps = 1000
    steps = 20
    env = DummyVecEnv([make_env(steps=steps, max_steps=max_steps, render_mode="human")])
    env = VecNormalize.load(os.path.join(model_dir, "vecnormalize.pkl"), env)
    env.training = False
    env.norm_reward = False

    model = load_model(model_dir=model_dir, env=env)
    obs = env.reset()
    total_reward = 0
    for _ in range(1020):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        if done[0]:
            break
    print(total_reward)

    env.close()


if __name__ == "__main__":
    train()
    # test()
