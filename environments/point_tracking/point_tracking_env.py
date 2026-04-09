import math, os, json, time
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from utils.tools import RAMathUtil
from communication.tcp_client import SimulationClient


class PointTrackingEnv(gym.Env):
    """
    点跟踪环境，用于与仿真平台交互 (Gymnasium版本)
    """

    def __init__(self, simulation_client, env_id, max_steps: int = 200, random_init: Optional[bool] = False,
                 render_mode: Optional[str] = None):
        """
        初始化环境

        Args:
            simulation_client: 仿真平台客户端
            max_steps: 每个episode的最大步数
        """
        super(PointTrackingEnv, self).__init__()

        self.simulation = simulation_client
        self.env_id = env_id
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

        # 目标位置（可以根据任务动态设置）经纬度
        self.target_position = None  # 经度、维度、高度等，目标点的信息->tacview格式

        # 定义动作空间：连续动作，控制点的移动 [x, y, z, 其他参数?]
        # 根据你的仿真平台调整维度
        # 升降舵，副翼，方向舵，油门
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
            shape=(23,),  # 根据实际观测维度调整
            dtype=np.float64
        )

        # 用于跟踪episode信息
        self.episode_reward = 0
        self.episode_length = 0

    def _process_observation(self) -> np.ndarray:
        """
        处理原始观测数据，转换为numpy数组

        提取点跟踪特征
        observation:{
            'data':
                {'0':
                    {'done': False,
                    'obs':
                        {'platforms':
                            [{  'alt': 7000.00022466667,
                                'beta': 4.285204235410325e-07,
                                'beta_rate': 8.514920699661757e-05,
                                'heading': 2.001073233123954e-12,
                                'lat': 35.00001347478865,
                                'lon': 117.00000000000345,
                                'mass': 3247.6942730385194,
                                'name': '1001',
                                'pitch': 1.3505474424370734e-05,
                                'pitch_rate': -0.0004268741334287518,
                                'roll': 2.8281869246863484e-12,
                                'roll_rate': 2.097094132424041e-06,
                                'side': 'Red',
                                'speed': 299.9806693962693,
                                'type': 'F-16',
                                'vx': 299.9806663253433,
                                'vy': 0.00012854785398021116,
                                'vz': 0.04292342442543375}],
                            'sim_time': 0.016666666666666666}}},
                            'msg': 'Batch Step Completed',
                            'req_id': 'step_1774698177753',
                            'status': 'ok'}

        return:
            [alt,
            beta_sin,
            beta_cos,
            beta_rate,
            heading,
            delta_heading,
            lat,
            lon,
            target_lat,
            target_lon,
            pitch_sin,
            pitch_cos,
            pitch_rate,
            roll_sin,
            roll_cos,
            roll_rate,
            speed,
            vx,
            vy,
            vz,
            delta_x,
            delta_y,
            delta_z]
        """

        plane_info = self.observation["data"][f"{self.env_id}"]["obs"]['platforms'][0]
        plane_info["target_lat"] = self.target_position["lat"]
        plane_info["target_lon"] = self.target_position["lon"]
        plane_info["target_alt"] = self.target_position["alt"]

        bearing = RAMathUtil.calculate_bearing(plane_info["lat"], plane_info["lon"], self.target_position["lat"],
                                               self.target_position["lon"])
        # 注意python取模是尽可能地让商更小
        delta_heading = (bearing - plane_info["heading"] + 180) % 360 - 180

        features = []
        # 飞机状态 23 维
        features.extend([
            plane_info.get("alt", 0) / 10000,
            np.sin(np.deg2rad(plane_info.get("beta", 0))),
            np.cos(np.deg2rad(plane_info.get("beta", 0))),
            np.deg2rad(plane_info.get("beta_rate", 0)),
            # heading:-180~180  0:north,90:east,-90:west,-180/180:south
            np.deg2rad(plane_info.get("heading", 0)),
            np.deg2rad(delta_heading),
            plane_info.get("lat", 0) / 90,
            plane_info.get("lon", 0) / 180,
            plane_info.get("target_lat", 0) / 90,
            plane_info.get("target_lon", 0) / 180,
            np.sin(np.deg2rad(plane_info.get("pitch", 0))),
            np.cos(np.deg2rad(plane_info.get("pitch", 0))),
            np.deg2rad(plane_info.get("pitch_rate", 0)),
            np.sin(np.deg2rad(plane_info.get("roll", 0))),
            np.cos(np.deg2rad(plane_info.get("roll", 0))),
            np.deg2rad(plane_info.get("roll_rate", 0)),
            plane_info.get("speed", 0) / 340,
            plane_info.get("vx", 0) / 340,
            plane_info.get("vy", 0) / 340,
            plane_info.get("vz", 0) / 340,
        ])

        delta_x, delta_y = RAMathUtil.convert_lat_long_to_xy(plane_info, self.target_position)
        delta_z = plane_info.get("alt", 0) - self.target_position['alt']
        delta_x /= 10000
        delta_y /= 10000
        delta_z /= 10000
        features.extend([delta_x, delta_y, delta_z])

        state = np.clip(features, self.observation_space.low, self.observation_space.high)
        return np.array(state)

    def _calculate_reward(self) -> float:
        """
        计算奖励值

        Args:
            observation: 当前观测
            state: 提取的特征
            action: 当前state下采取的动作

        Returns:
            奖励值

        """
        plane_info = self.observation["data"][f"{self.env_id}"]["obs"]['platforms'][0]
        delta_x, delta_y, delta_z = [v * 10000 for v in self.state[-3:]]
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

        delta_heading = self.state[5]  # 弧度
        roll = plane_info["roll"]
        pitch = plane_info["pitch"]
        roll_rate = plane_info.get("roll_rate", 0)
        pitch_rate = plane_info.get("pitch_rate", 0)
        beta_rate = plane_info.get("beta_rate", 0)
        beta = plane_info.get("beta", 0)
        speed = plane_info.get("speed", 0)
        alt = plane_info["alt"]
        vz = plane_info["vz"]

        task_reward = 0.0
        distance_reward = 0.0
        heading_reward = 0.0
        alt_reward = 0.0
        time_reward = -0.01
        speed_reward = 0.0
        angle_reward = 0.0

        # 任务奖励：任务成功给一个远大于单步奖励累计的正奖励；接近任务时给一些正奖励
        if distance <= 1000 and abs(delta_z) <= 100:
            task_reward += 1000

        # 距离奖励
        distance_reward += RAMathUtil.hyperbolic_function(x=distance, x_min=0, x_max=1000, lam=0.00001)

        # 方向奖励
        heading_reward += RAMathUtil.hyperbolic_function(x=delta_heading, x_min=-0.02, x_max=0.02, lam=0.01)

        # 高度奖励
        alt_reward += RAMathUtil.hyperbolic_function(x=delta_z, x_min=-100, x_max=100, lam=0.00005)
        if alt < 4000:
            # vz向上为负
            alt_reward += -np.clip(vz / 50 * (4000 - alt) / 4000, 0, 1)
        if alt < 3000:
            alt_reward += np.clip(alt / 3000, 0, 1) - 2
        if alt < 2000 or alt > 20000:
            alt_reward += -1000

        # 速度奖励
        speed_reward += RAMathUtil.hyperbolic_function(x=speed, x_min=100, x_max=600, lam=0.0002)
        if speed < 50 or speed > 800:
            speed_reward += -1000

        # 角度相关奖励
        angle_reward += RAMathUtil.hyperbolic_function(x=roll_rate, x_min=-120, x_max=120, lam=0.0001)
        angle_reward += RAMathUtil.hyperbolic_function(x=pitch_rate, x_min=-60, x_max=60, lam=0.0002)
        angle_reward += RAMathUtil.hyperbolic_function(x=roll, x_min=-60, x_max=60, lam=0.0002)
        angle_reward += RAMathUtil.hyperbolic_function(x=pitch, x_min=-60, x_max=60, lam=0.0002)
        angle_reward += RAMathUtil.hyperbolic_function(x=beta_rate, x_min=-5, x_max=5, lam=0.001)
        angle_reward += RAMathUtil.hyperbolic_function(x=beta, x_min=-5, x_max=5, lam=0.001)
        if abs(plane_info["beta"]) > 20:
            angle_reward += -1000

        total_reward = task_reward + distance_reward + heading_reward + alt_reward + time_reward + speed_reward + angle_reward
        return float(total_reward)

    def _check_terminated(self) -> bool:
        """
        检查是否终止（任务完成或失败）

        Args:
            observation: 当前观测
            state: 提取的特征

        Returns:
            bool: 是否终止
        """
        # 成功到达目标
        delta_x, delta_y, delta_z = [v * 10000 for v in self.state[-3:]]
        distance = math.sqrt(delta_x ** 2 + delta_y ** 2)

        if distance <= 1000 and abs(delta_z) <= 100:  # 到达目标的阈值
            print("task succeed")
            return True

        # 飞机超过或低于临界高度
        plane_info = self.observation["data"][f"{self.env_id}"]["obs"]['platforms'][0]
        if plane_info["alt"] < 2000.0 or plane_info["alt"] > 20000.0:
            if plane_info["alt"] > 20000.0:
                print("altitude too high")
            elif plane_info["alt"] < 2000.0:
                print("altitude too low")
            return True

        # 飞机超过或低于临界速度
        if plane_info["speed"] < 50 or plane_info["speed"] > 800:
            if plane_info["speed"] < 50:
                print("speed too low")
            elif plane_info["speed"] > 800:
                print("speed too high")
            return True

        # 飞机侧滑角过大
        if abs(plane_info["beta"]) > 20:
            print("beta too big")
            return True

        return False

    def _check_truncated(self) -> bool:
        """
        检查是否截断（时间耗尽）

        Returns:
            bool: 是否截断
        """
        # 达到最大步数
        if self.current_step >= self.max_steps:
            return True
        return False

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
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
        self.action_sequence.append(action)
        self.observation = self.simulation.get_environment_data([action.tolist()], env_id=self.env_id)
        self.observation_sequence.append(self.observation)

        # 处理观测
        self.state = self._process_observation()
        self.state_sequence.append(self.state)

        # 计算奖励
        reward = self._calculate_reward()

        # 检查是否终止和截断
        terminated = self._check_terminated()
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

        # 如果需要渲染
        if self.render_mode == "human":
            self.render()

        if terminated or truncated:
            print(self.episode_reward, self.episode_length)

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
        self.state = None
        self.target_position = None
        self.observation_sequence.clear()
        self.state_sequence.clear()
        self.action_sequence.clear()
        if self.render_mode == "human":
            self.reset_logs()

        # 重置步数和奖励
        self.current_step = 0
        self.episode_reward = 0
        self.episode_length = 0

        if options is not None:
            scenario = options.get("scenario", "testWzz")
            target_position = options.get("target_position", None)

        # 重置仿真
        try:
            if self.random_init:
                self._random_init()
            self.observation = self.simulation.reset(self.env_id)
        except Exception as e:
            # 返回零观测和错误信息
            info = {"error": str(e)}
            return np.zeros(self.observation_space.shape, dtype=np.float64), info

        # 设置新的目标位置
        if target_position is not None:
            self.target_position = target_position
        else:
            # 随机生成目标位置（可选）
            random_target_position = RAMathUtil.generate_target_arc(min_dist=20000, max_dist=30000)
            self.target_position = RAMathUtil.convert_xy_to_lat_long(
                self.observation["data"][f"{self.env_id}"]["obs"]['platforms'][0],
                random_target_position[0],
                random_target_position[1],
                random_target_position[2]
            )

        self.center_position = self.observation["data"][f"{self.env_id}"]["obs"]['platforms'][0]

        # 处理观测
        self.state = self._process_observation()

        self.observation_sequence.append(self.observation)
        self.state_sequence.append(self.state)

        # 构建信息字典
        error_info = {
            "initial_target_position": self.target_position.copy(),
            "initial_observation": self.observation
        }

        # 如果需要渲染
        if self.render_mode == "human":
            self.render()

        return self.state, error_info

    def close(self):
        """
        关闭环境
        """
        if hasattr(self, 'simulation') and self.simulation:
            self.simulation.close()

    def render(self, output_dir='logs', output_file='point_tracking.acmi'):
        """
        简化的render方法，精确匹配提供的ACMI格式
        """
        output_path = os.path.join(output_dir, output_file)

        if not hasattr(self, 'observation') or self.observation is None:
            return None

        try:
            # 解析数据
            if isinstance(self.observation, str):
                data = json.loads(self.observation)
            else:
                data = self.observation

            platforms = data.get('data', {}).get('0', {}).get('obs', {}).get('platforms', [])
            sim_time = data.get('data', {}).get('0', {}).get('obs', {}).get('sim_time', 0)

            if not platforms:
                return None

            # 初始化（如果是第一次调用）
            if not hasattr(self, '_base_time'):
                self._base_time = datetime.now()
                self._frame_count = 0
                self._platform_ids = {'1001': '5160'}  # 根据你的数据分配ID

            # 以追加模式打开文件
            with open(output_path, 'a', encoding='utf-8') as f:
                # 如果是新文件，写入头信息
                if self._frame_count == 0:
                    f.write("FileType=text/acmi/tacview\n")
                    f.write("FileVersion=2.2\n")
                    f.write(f"0,ReferenceTime={self._base_time.strftime('%Y-%m-%dT%H:%M:%S')}Z\n")

                # 写入时间戳
                f.write(f"#{sim_time:.2f}\n")

                # 为每个平台写入数据
                for i, platform in enumerate(platforms):
                    name = platform.get('name', '1001')

                    # 获取ID映射
                    if name in self._platform_ids:
                        object_id = self._platform_ids[name]
                    else:
                        object_id = str(5160 + i)
                        self._platform_ids[name] = object_id

                    # 获取数据
                    lat = platform.get('lat', 0)
                    lon = platform.get('lon', 0)
                    alt = platform.get('alt', 0)
                    roll = platform.get('roll', 0)
                    pitch = platform.get('pitch', 0)
                    heading = platform.get('heading', 0)

                    # 构建数据行
                    # 格式: ID,T=时间戳|经度|纬度|高度|滚转|俯仰|偏航,Name=名称,Type=类型,CallSign=呼号,Color=颜色
                    data_line = (f"{object_id},T={lon:.8f}|{lat:.8f}|{alt:.2f}|"
                                 f"{roll:.12f}|{pitch:.12f}|{heading:.6f},"
                                 f"Name=F-16,Type=Air+FixedWing,CallSign=F-16,Color=Red")

                    f.write(data_line + "\n")

                self._frame_count += 1

            return output_path

        except Exception as e:
            print(f"错误: {e}")
            return None

    def reset_logs(self, output_dir='logs', output_file='point_tracking.acmi'):
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建目录: {output_dir}")
        # 构建完整的输出文件路径
        output_path = os.path.join(output_dir, output_file)

        # 如果文件已存在，先删除
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"删除已存在的文件: {output_path}")

    def _write_target_to_log(self):
        target_position = {
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
        data_line = RAMathUtil.plane_to_encode(target_position)
        target_position = data_line
        with open(self.simulation.logger.output_path, 'a', encoding='utf-8') as f:
            f.write(target_position + "\n")

    def _random_init(self):
        """
        可以根据需求修改初始化(reset时调用)飞机信息
        """
        self.simulation.init_payload["initial_state"]["1001"]["heading"] = np.random.uniform(-1, 1) * 180
        # AFSim期望的pitch和roll都是[-90,90]
        self.simulation.init_payload["initial_state"]["1001"]["pitch"] = np.random.uniform(-1, 1) * 90
        self.simulation.init_payload["initial_state"]["1001"]["roll"] = np.random.uniform(-1, 1) * 90


def make_env(steps, max_steps):
    def _init():
        simulation = SimulationClient(
            env_num=1,
            host='127.0.0.1',
            port=8888,
            env_name="control",
            steps=steps,
            # log_save=True
        )
        simulation.connection()

        env = PointTrackingEnv(
            simulation_client=simulation,
            env_id=0,  # 单环境内部永远是0
            max_steps=max_steps
        )
        return Monitor(env)

    return _init


# 使用示例
if __name__ == "__main__":

    # 1. 创建环境（Gymnasium版本）
    env_num = 10
    max_steps = 1020
    steps = 20
    env = SubprocVecEnv([make_env(steps=steps, max_steps=max_steps) for _ in range(env_num)])

    # reset（VecEnv返回的是obs数组）
    obs = env.reset()
    print("reset obs:", obs.shape)  # (env_num, obs_dim)

    start_time = time.time()
    for i in range(math.ceil(max_steps / env_num)):
        # 给所有环境同一个动作
        action = np.array([[0.0, 0.0, 0.0, 0.5]] * env_num)

        obs, rewards, dones, infos = env.step(action)

        # print(f"\nStep {i}")
        # print("obs shape:", obs.shape)
        # print("rewards:", rewards)
        # print("dones:", dones)
    end_time = time.time()
    print("总耗时:", end_time - start_time, "秒")
    print("平均每步:", (end_time - start_time) / (math.ceil(max_steps / env_num) * env_num), "秒")
    print("FPS:", (math.ceil(max_steps / env_num) * env_num) / (end_time - start_time))
    env.close()