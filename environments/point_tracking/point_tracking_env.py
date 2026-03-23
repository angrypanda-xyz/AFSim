import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from utils.tools import RAMathUtil
import math, os, json
from communication.tcp_client import SimulationClient


class PointTrackingEnv(gym.Env):
    """
    点跟踪环境，用于与仿真平台交互 (Gymnasium版本)
    """

    def __init__(self, simulation_client, max_steps: int = 200, render_mode: Optional[str] = None, random_init: Optional[bool] = False):
        """
        初始化环境

        Args:
            simulation_client: 仿真平台客户端
            max_steps: 每个episode的最大步数
            render_mode: 渲染模式，可选'human'或None
        """
        super(PointTrackingEnv, self).__init__()

        self.simulation = simulation_client
        self.simulation.connection()
        self.max_steps = max_steps
        self.random_init = random_init
        self.render_mode = render_mode
        self.current_step = 0
        self.observation = None
        self.state = None
        self.target_position = np.array([12000.0, 0.0, 0.0])  # x, y, z
        self.target_point = None  # 经度、维度、高度等，目标点的信息->tacview格式
        self.center_position = {'alt': 0.0, 'lat': 0.0, 'lon': 0.0}  # 中心点的经、纬、高度，默认是本机中心点
        self.observation_sequence = []
        self.state_sequence = []
        self.action_sequence = []

        # 定义动作空间：连续动作，控制点的移动 [x, y, z, 其他参数?]
        # 根据你的仿真平台调整维度
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),  # 最小控制值
            high=np.array([1.0, 1.0, 1.0, 1.0]),  # 最大控制值
            shape=(4,),
            dtype=np.float64
        )

        # 定义观测空间：环境返回的数据
        # 这里需要根据simulation.get_environment_data返回的实际结构调整
        # 假设观测是包含位置、速度等信息的向量
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),  # 根据实际观测维度调整
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
        delta_x, delta_y = RAMathUtil.convert_lat_long_to_xy(plane_info, self.target_position)
        delta_z = self.target_position["alt"] - plane_info["alt"]

        state = np.array([delta_x / 15000, delta_y / 15000, delta_z / 15000,
                          plane_info["heading"] / 180,
                          plane_info["beta"] / 180, plane_info["beta_rate"] / 180,
                          plane_info["pitch"] / 90, plane_info["pitch_rate"] / 180,
                          plane_info["roll"] / 90, plane_info["roll_rate"] / 180,
                          (plane_info["speed"] - 200) / 200,
                          plane_info["vx"] / 400, plane_info["vy"] / 400, plane_info["vz"] / 400
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
        # 提取当前位置（假设前3个元素是位置）
        delta_x, delta_y, delta_z = self.state[0] * 15000, self.state[1] * 15000, self.state[2] * 15000

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
        delta_x, delta_y, delta_z = self.state[0] * 15000, self.state[1] * 15000, self.state[2] * 15000
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

        if self.simulation.log_save:
            """如果飞机数据被保存了，这里将目标点一起写入到文件中"""
            self._write_target_to_log()
        # 确保动作在合法范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        for i in range(slice):
            # 连续多少帧再重新生成一个新的动作
            self.observation = self.simulation.get_environment_data([action.tolist()])
            self.observation['data']['0']['obs']['platforms'][0]["target_position"] = self.target_position
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
        scenario = "testWzz"
        target_position = None
        self.observation = None
        if options is not None:
            scenario = options.get("scenario", "testWzz")
            target_position = options.get("target_position", None)

        # 重置仿真
        try:
            if self.random_init:
                self._random_init()
            self.observation = self.simulation.reset()
        except Exception as e:
            # 返回零观测和错误信息
            error_info = {"error": str(e)}
            return np.zeros(self.observation_space.shape, dtype=np.float64), error_info

        # 设置新的目标位置
        if target_position is not None:
            self.target_position = np.array(target_position, dtype=np.float64)
        else:
            # 随机生成目标位置（可选）
            random_target_position = RAMathUtil.generate_target_arc()
            self.target_position = RAMathUtil.convert_xy_to_lat_long(
                self.observation["data"]["0"]["obs"]['platforms'][0],
                random_target_position[0],
                random_target_position[1],
                delta_z=random_target_position[2]
            )
            self.observation['data']['0']['obs']['platforms'][0]["target_position"] = self.target_position

        # 重置步数和奖励
        self.current_step = 0
        self.episode_reward = 0
        self.episode_length = 0

        self.center_position = self.observation["data"]["0"]["obs"]['platforms'][0]

        # 处理观测
        self.state = self._process_observation()

        # 构建信息字典
        error_info = {
            "initial_target_position": self.target_position.copy(),
            "initial_observation": self.observation
        }

        return self.state, error_info

    def close(self):
        """
        关闭环境
        """
        if hasattr(self, 'simulation') and self.simulation:
            self.simulation.close()

    def _write_target_to_log(self):
        if self.target_point is None:
            # print(self.target_position)
            # target_lat_lon_alt = RAMathUtil.convert_xy_to_lat_long(self.center_position,
            #                                                        self.target_position[0],
            #                                                        self.target_position[1],
            #                                                        self.target_position[2])
            target_point = {
                # 'lat': target_lat_lon_alt['lat'],
                'lat': self.target_position['lat'],
                # 'lon': target_lat_lon_alt['lon'],
                'lon': self.target_position['lon'],
                # 'alt': target_lat_lon_alt['alt'],
                'alt': self.target_position['alt'],
                'name': '10001',
                'heading': 0,
                'pitch': 0.0,
                'roll': 0.0,
                'speed': 0.0,
                'side': 'Blue',
                'type': 'Point'
            }
            data_line = RAMathUtil.plane_to_encode(target_point)
            self.target_point = data_line
        with open(self.simulation.logger.output_path, 'a', encoding='utf-8') as f:
            f.write(self.target_point + "\n")

    def _random_init(self):
        """
        可以根据需求修改初始化(reset时调用)飞机信息
        """
        self.simulation.init_payload["initial_state"]["1001"]["heading"] = np.random.uniform(-1, 1) * 180
        # AFSim期望的pitch和roll都是[-90,90]
        self.simulation.init_payload["initial_state"]["1001"]["pitch"] = np.random.uniform(-1, 1) * 90
        self.simulation.init_payload["initial_state"]["1001"]["roll"] = np.random.uniform(-1, 1) * 90


# 使用示例
if __name__ == "__main__":
    # 1. 创建环境（Gymnasium版本）
    simulation = SimulationClient(host='127.0.0.1', port=8888, steps=1, env_name="control", log_save=True)
    env = PointTrackingEnv(simulation_client=simulation, max_steps=2000, render_mode=None, random_init=True)
    # 重置环境，现在返回两个值
    state, info = env.reset()
    for i in range(2000):
        action = np.array([0.5, 0.0, 0.0, 1.0])
        # Gymnasium的step返回5个值
        state, reward, terminated, truncated, info = env.step(action)
        # 检查是否结束（终止或截断）
        done = terminated or truncated

        if done:
            print(f"Episode ended: reward={reward}, terminated={terminated}, truncated={truncated}")
            # state, info = env.reset()
            break
    env.close()
