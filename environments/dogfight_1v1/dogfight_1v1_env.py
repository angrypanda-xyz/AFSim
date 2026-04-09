import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from utils.tools import RAMathUtil
import math, os, json
from communication.tcp_client import SimulationClient


class DogFight1v1Env(gym.Env):
    """
    1v1狗斗环境，用于与仿真平台交互 (Gymnasium版本)
    """

    def __init__(self, simulation_client, max_steps: int = 200, render_mode: Optional[str] = None, random_init: Optional[bool] = False):
        """
        初始化环境

        Args:
            simulation_client: 仿真平台客户端
            max_steps: 每个episode的最大步数
            render_mode: 渲染模式，可选'human'或None
        """
        super(DogFight1v1Env, self).__init__()

        self.simulation = simulation_client
        self.simulation.connection()
        self.max_steps = max_steps
        self.random_init = random_init
        self.render_mode = render_mode
        self.current_step = 0
        self.observation = None
        self.state = None
        self.center_position = {'alt': 0.0, 'lat': 0.0, 'lon': 0.0}  # 中心点的经、纬、高度，默认是本机中心点
        self.observation_sequence = []
        self.state_sequence = []
        self.action_sequence = []

        # 定义动作空间：连续动作，控制点的移动 [x, y, z, 其他参数?]
        # 根据你的仿真平台调整维度
        # 暂不将导弹发射逻辑添加到强化学习训练中，达到发射条件就发射导弹
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),  # 最小控制值
            high=np.array([1.0, 1.0, 1.0, 1.0]),  # 最大控制值
            shape=(4,),
            dtype=np.float64
        )
        # actions = [
        #     [0.5, 0.0, 0.0, 1.0, False, "5001"],
        #     [0.5, 0.0, 0.0, 1.0, False, "1001"]
        # ]

        # 定义观测空间：环境返回的数据
        # 这里需要根据simulation.get_environment_data返回的实际结构调整
        # 假设观测是包含位置、速度等信息的向量
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(28,),  # 根据实际观测维度调整
            dtype=np.float64
        )

        # 用于跟踪episode信息
        self.episode_reward = 0
        self.episode_length = 0

        # Gymnasium的metadata格式
        self.metadata = {"render_modes": ["human"], "render_fps": 30}

    def _process_observation(self) -> np.ndarray:
        """
        处理原始观测数据，转换为numpy数组

        Args:
            raw_observation: 仿真平台返回的原始观测

        Returns:
            处理后的观测数组
        """
        plane_info = self.observation["data"]["0"]["obs"]['platforms'][0]
        enemy_info = self.observation["data"]["0"]["obs"]['platforms'][1]
        plane_x, plane_y, plane_z = RAMathUtil.convert_aircraft_xyz(plane_info, self.center_position)
        enemy_x, enemy_y, enemy_z = RAMathUtil.convert_aircraft_xyz(enemy_info, self.center_position)
        state = np.array(
            [
                plane_x / 15000, plane_y / 15000, plane_z / 15000,
                enemy_x / 15000, enemy_y / 15000, enemy_z / 15000,
                plane_info["heading"] / 180, (plane_info["speed"] - 200) / 200,
                plane_info["beta"] / 180, plane_info["beta_rate"] / 180,
                plane_info["pitch"] / 90, plane_info["pitch_rate"] / 180,
                plane_info["roll"] / 90, plane_info["roll_rate"] / 180,
                plane_info["vx"] / 400, plane_info["vy"] / 400, plane_info["vz"] / 400,
                enemy_info["heading"] / 180, (enemy_info["speed"] - 200) / 200,
                enemy_info["beta"] / 180, enemy_info["beta_rate"] / 180,
                enemy_info["pitch"] / 180, enemy_info["pitch_rate"] / 180,
                enemy_info["roll"] / 180, enemy_info["roll_rate"] / 180,
                enemy_info["vx"] / 400, enemy_info["vy"] / 400, enemy_info["vz"] / 400,
            ], dtype=np.float64)
        return np.array(state)

    def _calculate_reward(self) -> float:
        """
        计算奖励值

        Args:
            observation: 当前观测

        Returns:
            奖励值
        """
        plane_info = self.observation["data"]["0"]["obs"]['platforms'][0]
        enemy_info = self.observation["data"]["0"]["obs"]['platforms'][1]
        plane_x, plane_y, plane_z = RAMathUtil.convert_aircraft_xyz(plane_info, self.center_position)
        enemy_x, enemy_y, enemy_z = RAMathUtil.convert_aircraft_xyz(enemy_info, self.center_position)

        # 提取当前位置（假设前3个元素是位置）
        delta_x, delta_y, delta_z = plane_x - enemy_x, plane_y - enemy_y, plane_z - enemy_z
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

        # 奖励函数设计
        # 1. 距离惩罚（负奖励）
        distance_penalty = -distance * 0.00002

        # 2. 成功到达目标的奖励
        success_reward = 10.0 if distance < 1000 else 0.0

        # 3. 时间惩罚（鼓励快速到达）
        time_penalty = -0.002

        # 4. 平滑性奖励（可选，需要速度信息）
        # velocity = observation[3:6]
        # smooth_penalty = -0.001 * np.linalg.norm(velocity)

        # 5.超出高度限制，判定飞机坠毁
        plane_info = self.observation["data"]["0"]["obs"]['platforms'][0]
        if plane_info['alt'] < 1000.0:
            return float(-10.0)

        total_reward = distance_penalty + success_reward + time_penalty

        return float(total_reward)

    def _check_terminated(self) -> bool:
        """
        检查是否终止（任务完成）

        Args:
            observation: 当前观测

        Returns:
            bool: 是否终止
        """
        # 成功到达目标
        plane_info = self.observation["data"]["0"]["obs"]['platforms'][0]
        enemy_info = self.observation["data"]["0"]["obs"]['platforms'][1]
        plane_x, plane_y, plane_z = RAMathUtil.convert_aircraft_xyz(plane_info, self.center_position)
        enemy_x, enemy_y, enemy_z = RAMathUtil.convert_aircraft_xyz(enemy_info, self.center_position)

        # 提取当前位置（假设前3个元素是位置）
        delta_x, delta_y, delta_z = plane_x - enemy_x, plane_y - enemy_y, plane_z - enemy_z
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2 + delta_z ** 2)

        if distance < 500:  # 到达目标的阈值
            return True

        # 飞机坠毁
        plane_info = self.observation["data"]["0"]["obs"]['platforms'][0]
        if plane_info['alt'] < 1000.0:
            return True

        return False

    def _check_truncated(self) -> bool:
        """
        检查是否截断（时间耗尽等）

        Args:
            observation: 当前观测

        Returns:
            bool: 是否截断
        """
        # 达到最大步数
        if self.current_step >= self.max_steps:
            return True

        return False

    def step(self, action: np.ndarray, slice: int = 1) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 动作向量
            slice: 每传入一个动作，这里推进几次
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # 确保动作在合法范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        """这里要传两个飞机的动作，给蓝机一个默认动作"""
        actions = [
            [action[0], action[1], action[2], action[3], False, "5001"],  # 后续要增加导弹发射逻辑
            [-0.05, 0.0, 0.0, 0.5, False, "1001"]
        ]
        for i in range(slice):
            # 连续多少帧再重新生成一个新的动作
            # self.observation = self.simulation.get_environment_data([action.tolist()])
            self.observation = self.simulation.get_environment_data(actions)
            self.observation_sequence.append(self.observation)
        # 处理观测
        self.state = self._process_observation()
        self.state_sequence.append(self.state)

        # 计算奖励
        reward = self._calculate_reward()

        # 检查是否终止和截断
        terminated = self._check_terminated()
        if terminated:
            print(reward)
        truncated = self._check_truncated()

        # 更新步数
        self.current_step += 1
        self.episode_reward += reward
        self.episode_length += 1

        # 信息字典
        info = {
            'episode': {
                'r': self.episode_reward,
                'l': self.episode_length
            },
        }

        return self.state, reward, terminated, truncated, info

    def reset(self,
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境

        Args:
            seed: 随机种子
            options: 重置选项，可以包含scenario和target_position

        Returns:
            tuple: (observation, info)
        """
        # 设置随机种子
        super().reset(seed=seed)

        # 从options中获取参数
        scenario = "test1v1"
        target_position = None
        self.observation = None
        if options is not None:
            scenario = options.get("scenario", "test1v1")

        # 重置仿真
        try:
            if self.random_init:
                """双方位置要增加一个随机性"""
                self._random_init()

            self.observation = self.simulation.reset()
        except Exception as e:
            # 返回零观测和错误信息
            error_info = {"error": str(e)}
            return np.zeros(self.observation_space.shape, dtype=np.float64), error_info

        # 重置步数和奖励
        self.current_step = 0
        self.episode_reward = 0
        self.episode_length = 0

        # 以红机位置未中心点建立坐标系
        self.center_position = self.observation["data"]["0"]["obs"]['platforms'][0]

        # 处理观测
        self.state = self._process_observation()

        # 构建信息字典
        error_info = {
            "initial_observation": self.observation
        }

        return self.state, error_info

    def close(self):
        """
        关闭环境
        """
        if hasattr(self, 'simulation') and self.simulation:
            self.simulation.close()

    def _random_init(self):
        """
        可以根据需求修改初始化(reset时调用)飞机信息
        """
        self.simulation.init_payload["initial_state"]["1001"]["heading"] = np.random.uniform(-1, 1) * 180
        # AFSim期望的pitch和roll都是[-90,90]
        # self.simulation.init_payload["initial_state"]["1001"]["pitch"] = np.random.uniform(-1, 1) * 90
        # self.simulation.init_payload["initial_state"]["1001"]["roll"] = np.random.uniform(-1, 1) * 90

        self.simulation.init_payload["initial_state"]["5001"]["heading"] = np.random.uniform(-1, 1) * 180
        # AFSim期望的pitch和roll都是[-90,90]
        # self.simulation.init_payload["initial_state"]["5001"]["pitch"] = np.random.uniform(-1, 1) * 90
        # self.simulation.init_payload["initial_state"]["5001"]["roll"] = np.random.uniform(-1, 1) * 90


# 使用示例
if __name__ == "__main__":
    # 1. 创建环境（Gymnasium版本）
    simulation = SimulationClient(host='127.0.0.1', port=8888, steps=10, environment="dogfight", log_save=True)
    env = DogFight1v1Env(simulation_client=simulation, max_steps=2000, render_mode=None, random_init=True)
    # 重置环境，现在返回两个值
    state, info = env.reset()
    for i in range(2000):
        action = np.array([-0.05, 0.0, 0.0, 0.5])
        # Gymnasium的step返回5个值
        state, reward, terminated, truncated, info = env.step(action)
        # 检查是否结束（终止或截断）
        done = terminated or truncated

        if done:
            print(f"Episode ended: reward={reward}, terminated={terminated}, truncated={truncated}")
            # state, info = env.reset()
            break
    env.close()
