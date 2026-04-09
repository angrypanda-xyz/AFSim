import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from utils.tools import RAMathUtil


class TacViewLogger:
    """
    Tacview ACMI 文件记录器
    用于生成 Tacview 可识别的飞行数据记录文件
    """

    # 类常量
    DEFAULT_OUTPUT_DIR = 'logs'
    DEFAULT_OUTPUT_FILE = 'fighter'
    REQUIRED_PLANE_FIELDS = {'name', 'side', 'lat', 'lon', 'alt', 'roll', 'pitch', 'heading'}

    def __init__(self, output_dir: str = DEFAULT_OUTPUT_DIR, output_file: str = DEFAULT_OUTPUT_FILE, env_id=0):
        """
        初始化日志记录器

        Args:
            output_dir: 输出目录
            output_file: 输出文件名
        """
        self.output_dir = output_dir
        self.output_file = output_file+"_"+str(env_id)+".acmi"
        self.output_path = self._get_output_path()

        # 设置日志
        self._setup_logging()

        # 初始化文件
        self._initialize_file()

    def _get_output_path(self) -> str:
        """获取完整的输出文件路径"""
        current_dir = os.path.join(os.getcwd(), self.output_dir)
        return os.path.join(current_dir, self.output_file)

    def _setup_logging(self) -> None:
        """设置日志记录"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def _initialize_file(self) -> None:
        """初始化日志文件"""
        try:
            # 确保目录存在
            os.makedirs(self.output_dir, exist_ok=True)
            # self.logger.info(f"确保目录存在: {self.output_dir}")

            # 如果文件存在，先删除
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
                # self.logger.info(f"删除已存在的文件: {self.output_path}")

            # 写入文件头
            self._write_header()

        except OSError as e:
            self.logger.error(f"初始化文件失败: {e}")
            raise

    def _write_header(self) -> None:
        """写入ACMI文件头"""
        try:
            with open(self.output_path, 'w', encoding='utf-8') as f:
                f.write("FileType=text/acmi/tacview\n")
                f.write("FileVersion=2.2\n")

                # 写入参考时间（UTC）
                current_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
                f.write(f"0,ReferenceTime={current_time}Z\n")

            # self.logger.info(f"文件头写入成功: {self.output_path}")

        except IOError as e:
            self.logger.error(f"写入文件头失败: {e}")
            raise

    def _validate_plane_data(self, plane: Dict[str, Any]) -> bool:
        """
        验证飞机数据是否完整

        Args:
            plane: 飞机数据字典

        Returns:
            bool: 数据是否有效
        """
        missing_fields = self.REQUIRED_PLANE_FIELDS - set(plane.keys())

        if missing_fields:
            self.logger.warning(f"飞机数据缺少必要字段: {missing_fields}")
            return False

        return True

    def _format_plane_line(self, plane: Dict[str, Any]) -> Optional[str]:
        """
        格式化单架飞机的数据行

        Args:
            plane: 飞机数据字典

        Returns:
            Optional[str]: 格式化后的数据行，如果数据无效则返回None
        """
        if not self._validate_plane_data(plane):
            return None
        try:
            data_line = RAMathUtil.plane_to_encode(plane)
            return data_line
        except (KeyError, ValueError, TypeError) as e:
            self.logger.error(f"格式化飞机数据时出错: {e}, 数据: {plane}")
            return None

    def add(self, observation: Dict[str, Any]) -> None:
        """
        添加观测数据到日志文件

        Args:
            observation: 观测数据，包含 sim_time 和 platforms 信息
        """
        try:
            sim_time = observation["sim_time"]
            platforms = observation["platforms"]

            if not platforms:
                self.logger.debug("没有平台数据可记录")
                return

            # 写入时间戳
            with open(self.output_path, 'a', encoding='utf-8') as f:
                f.write(f"#{sim_time:.2f}\n")

                # 写入每个飞机的数据
                for plane in platforms:
                    formatted_line = self._format_plane_line(plane)
                    if formatted_line:
                        f.write(formatted_line)

            self.logger.debug(f"成功添加观测数据，时间: {sim_time:.2f}")

        except IOError as e:
            self.logger.error(f"写入观测数据失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"处理观测数据时发生未知错误: {e}")
            raise

    def reset(self) -> None:
        """重置日志文件"""
        self.logger.info("重置日志文件...")
        self._initialize_file()

    @property
    def file_info(self) -> Dict[str, Any]:
        """获取文件信息"""
        return {
            'path': self.output_path,
            'exists': os.path.exists(self.output_path),
            'size': os.path.getsize(self.output_path) if os.path.exists(self.output_path) else 0
        }


# 使用示例
if __name__ == "__main__":
    # 创建日志记录器
    logger = TacViewLogger(output_dir='logs', output_file='fighter.acmi')

    # 模拟数据
    test_observation = {
        "sim_time": 10.5,
        "platforms": [
            {
                "name": "1001",
                "side": "Blue",
                "lat": 30.12345678,
                "lon": 120.12345678,
                "alt": 5000.0,
                "roll": 0.1,
                "pitch": 0.2,
                "heading": 45.0
            },
            {
                "name": "2001",
                "side": "Red",
                "lat": 30.22345678,
                "lon": 120.22345678,
                "alt": 5100.0,
                "roll": -0.1,
                "pitch": 0.15,
                "heading": 90.0
            }
        ]
    }

    # 添加数据
    logger.add(test_observation)

    # 打印文件信息
    print(f"文件信息: {logger.file_info}")
