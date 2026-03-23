from typing import Dict, Type
from communication.tcp_client import SimulationClient

from aircraft_control.aircraft_control_env import AircraftControlEnv
from dogfight_1v1.dogfight_1v1_env import DogFight1v1Env
from point_tracking.point_tracking_env import PointTrackingEnv


class EnvironmentFactory:
    """环境工厂类"""

    # 环境名称到环境类的映射
    _environment_registry: Dict[str, Type] = {
        "aircraft_control": AircraftControlEnv,
        "point_tracking": PointTrackingEnv,
        "dogfight_1v1": DogFight1v1Env,
    }

    @classmethod
    def create_environment(cls,
                           env_name: str,
                           sim_client: SimulationClient,
                           render: bool = False,
                           save_acmi: bool = False,
                           acmi_file_path: str = None):
        """创建指定环境"""
        if env_name not in cls._environment_registry:
            available_envs = list(cls._environment_registry.keys())
            raise ValueError(f"环境 '{env_name}' 不存在。可用环境: {available_envs}")

        env_class = cls._environment_registry[env_name]
        return env_class(
            sim_client=sim_client,
            render=render,
            save_acmi=save_acmi,
            acmi_file_path=acmi_file_path
        )

    @classmethod
    def register_environment(cls, env_name: str, env_class: Type):
        """注册新环境"""
        cls._environment_registry[env_name] = env_class

    @classmethod
    def get_available_environments(cls) -> list:
        """获取可用环境列表"""
        return list(cls._environment_registry.keys())