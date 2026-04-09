import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict
from utils.tools import RAMathUtil
from communication.tcp_client import SimulationClient


def cycle_target_reset(current_value, target_value, cycle_value=2):
    return find_closest(current_value, target_value, cycle_value)


def find_closest(value, target, cycle_value=2):
    """
    计算value距离target-cycle_value、target、target+cycle_value哪个最近，返回最近的那个数
    """
    candidates = [target - cycle_value, target, target + cycle_value]
    return min(candidates, key=lambda x: abs(x - value))


def calculate_approach_reward(previous_value, current_value, target_value,
                              reward_factor=1.0, overshoot_penalty=2.0):
    """
    计算基于接近目标的奖励函数

    Args:
        previous_value: 前一时刻的值
        current_value: 当前时刻的值
        target_value: 目标值
        reward_factor: 正常接近时的奖励系数（正数）
        overshoot_penalty: 越过目标时的惩罚系数（正数，最终惩罚为负值）

    Returns:
        float: 奖励值
    """
    # 计算上一时刻和当前时刻到目标值的距离
    prev_distance = abs(previous_value - target_value)
    curr_distance = abs(current_value - target_value)

    # 情况1: 当前值比前一时刻更接近目标（距离减小）
    if curr_distance < prev_distance:
        # 检查是否穿过了目标（即目标值介于前后值之间）
        # 判断条件: (previous_value - target_value) * (current_value - target_value) < 0
        if (previous_value - target_value) * (current_value - target_value) < 0:
            # 穿过了目标值，给予惩罚
            return -overshoot_penalty
        else:
            # 正常接近，给予正奖励（奖励大小与接近程度成正比）
            improvement = (prev_distance - curr_distance) / (prev_distance + 1e-6)
            return reward_factor * improvement

    # 情况2: 当前值比前一时刻更远离目标（距离增大）
    elif curr_distance > prev_distance:
        # 给予负奖励，惩罚远离目标的行为
        degradation = (curr_distance - prev_distance) / (prev_distance + 1e-6)
        return -reward_factor * degradation

    # 情况3: 距离不变
    else:
        return 0.0


class AircraftControlEnv(gym.Env):
    """
    点跟踪环境，用于与仿真平台交互 (Gymnasium版本)
    """

    def __init__(self, simulation_client, max_steps: int = 200, render_mode: Optional[str] = None,
                 random_init: Optional[bool] = False):
        """
        初始化环境

        Args:
            simulation_client: 仿真平台客户端
            max_steps: 每个episode的最大步数
            render_mode: 渲染模式，"human"就是实时可视化
        """
        super(AircraftControlEnv, self).__init__()

        self.simulation = simulation_client
        self.simulation.connection()
        self.max_steps = max_steps
        self.random_init = random_init
        self.render_mode = render_mode
        self.current_step = 0
        self.observation = None
        self.state = None
        self.target_attitude = None
        self.target_plane = None
        self.observation_sequence = []
        self.state_sequence = []
        self.action_sequence = []
        self.keep_attitude_count = 0

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
            shape=(18,),  # 根据实际观测维度调整
            dtype=np.float64
        )

        # 目标姿态信息（可以根据任务动态设置）
        # self.target_attitude = np.array([0.0, 0.0, 0.0, 0.0])  # 俯仰角、滚转角、朝向角、速度->归一化[-1, 1]

        # 用于跟踪episode信息
        self.episode_reward = 0
        self.episode_length = 0

        # Gymnasium的metadata格式
        self.metadata = {"render_modes": ["human"], "render_fps": 30}

    def _process_observation(self) -> np.ndarray:
        """
        处理原始观测数据，转换为numpy数组，参数需要归一化

        Args:
            raw_observation: 仿真平台返回的原始观测

        Returns:
            处理后的观测数组
        """
        plane_info = self.observation["data"]["0"]["obs"]['platforms'][0]
        heading = plane_info["heading"] / 180
        pitch = plane_info["pitch"] / 90  # [-90,90] -> [-1,1]
        pitch_rate = plane_info["pitch_rate"] / 180
        roll = plane_info["roll"] / 90  # [-90,90] -> [-1,1]
        roll_rate = plane_info["roll_rate"] / 180
        speed = plane_info["speed"] / 400  # [0,400] -> [0,1]
        alt = plane_info["alt"] / 10000  # [0,10000] -> [0,1]
        beta = plane_info["beta"] / 180
        beta_rate = plane_info["beta_rate"] / 180
        mass = plane_info["mass"] / 4000  # 油量，暂无影响
        vx = plane_info["vx"] / 400
        vy = plane_info["vy"] / 400
        vz = plane_info["vz"] / 400
        target_heading = cycle_target_reset(heading, plane_info["target_attitude"][0])
        target_pitch = cycle_target_reset(pitch, plane_info["target_attitude"][1])
        target_roll = cycle_target_reset(roll, plane_info["target_attitude"][2])
        target_speed = plane_info["target_attitude"][3]

        return np.array([heading, pitch, roll, speed, alt, beta,
                         pitch_rate, roll_rate, beta_rate, mass, vx, vy, vz,
                         target_heading, target_pitch, target_roll, target_speed,
                         self.keep_attitude_count],
                        dtype=np.float64)

    def _calculate_reward(self) -> float:
        """
        计算奖励值
        Returns:
            奖励值
        """
        current_state = self.state_sequence[-1]
        previous_state = self.state_sequence[-2]
        # reward_heading = -(current_state[0] - current_state[13]) ** 2
        reward_heading = calculate_approach_reward(previous_state[0], current_state[0], previous_state[13])
        # reward_pitch = -(current_state[1] - current_state[14]) ** 2
        reward_pitch = calculate_approach_reward(previous_state[1], current_state[1], previous_state[14])
        # reward_roll = -(current_state[2] - current_state[15]) ** 2
        reward_roll = calculate_approach_reward(previous_state[2], current_state[2], previous_state[15])
        # reward_speed = -(current_state[3] - current_state[16]) ** 2
        reward_speed = calculate_approach_reward(previous_state[3], current_state[3], previous_state[16])

        # total_reward = 0.7 * reward_pitch + 0.1 * reward_roll + reward_heading + 0.3 * reward_speed
        total_reward = reward_heading + 0.5 * reward_pitch + 0.05 * reward_speed
        if reward_heading > -0.0001 and reward_pitch > -0.0001:
            self.keep_attitude_count += 0.1
        else:
            self.keep_attitude_count = 0
        if self.keep_attitude_count >= 0.5:
            total_reward = 100.0
        return float(total_reward)/50

    def _check_terminated(self) -> bool:
        """
        检查是否终止（任务完成）
        Returns:
            bool: 是否终止
        """
        # 成功到达目标
        current_state = self.state_sequence[-1]
        reward_heading = -(current_state[0] - current_state[13]) ** 2
        reward_pitch = -(current_state[1] - current_state[14]) ** 2
        if reward_heading > -0.0001 and reward_pitch > -0.0001:
            return True
        return False

    def _check_truncated(self) -> bool:
        """
        检查是否截断（时间耗尽等）
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

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # 确保动作在合法范围内
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        if self.simulation.log_save:
            """如果飞机数据被保存了，这里将目标点一起写入到文件中"""
            self._write_target_to_log()
        self.action_sequence.append(action)
        for i in range(slice):
            # 连续多少帧再重新生成一个新的动作
            self.observation = self.simulation.get_environment_data([action.tolist()])
            self.observation['data']['0']['obs']['platforms'][0]["target_attitude"] = np.array(self.target_attitude,
                                                                                               dtype=np.float64)
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

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None, ) -> Tuple[np.ndarray, Dict]:
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
        self.keep_attitude_count = 0
        target_attitude = None
        self.observation = None
        if options is not None:
            scenario = options.get("scenario", "testWzz")
            target_attitude = options.get("target_attitude", None)

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
        if target_attitude is not None:
            self.observation['data']['0']['obs']['platforms'][0]["target_attitude"] = np.array(target_attitude,
                                                                                               dtype=np.float64)
        else:
            # 随机生成目标位置（可选）
            self.target_attitude = RAMathUtil.generate_target_attitude()  # [heading, pitch, roll, speed]
            self.observation['data']['0']['obs']['platforms'][0]["target_attitude"] = self.target_attitude

            # self.target_attitude = random_target_attitude

        # 重置步数和奖励
        self.current_step = 0
        self.episode_reward = 0
        self.episode_length = 0

        # 处理观测
        self.state = self._process_observation()
        self.state_sequence.append(self.state)
        self.observation_sequence.append(self.observation)

        # 构建信息字典
        error_info = {
            "initial_target_attitude": self.observation['data']['0']['obs']['platforms'][0]["target_attitude"].copy(),
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
        if self.target_plane is None:
            plane = self.observation['data']['0']['obs']['platforms'][0]
            target_plane = {
                'lat': plane['lat'],
                'lon': plane['lon'],
                'alt': plane['alt'] - 1000,
                'name': '1002',
                'heading': plane["target_attitude"][0] * 180,
                'pitch': plane["target_attitude"][1] * 90,  # 取值范围[-90, 90]
                'roll': plane["target_attitude"][2] * 90,  # 取值范围[-90, 90]
                'speed': plane["target_attitude"][3] * 400,  # 取值范围[0, 400]
                'side': plane['side'],
                'type': plane['type']
            }
            data_line = RAMathUtil.plane_to_encode(target_plane)
            self.target_plane = data_line
        with open(self.simulation.logger.output_path, 'a', encoding='utf-8') as f:
            f.write(self.target_plane + "")

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
    simulation = SimulationClient(host='127.0.0.1', port=8888, steps=6, env_name="control", log_save=True)
    env = AircraftControlEnv(simulation_client=simulation, max_steps=100, render_mode=None, random_init=True)
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
