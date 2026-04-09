import socket
import json
import struct
import time
import uuid
import traceback
from communication.init_params import InitClass
from visualization.tacview_handler import TacView
from visualization.logger_handler import TacViewLogger


def check_response(resp):
    if resp is None:
        raise ConnectionError(f"初始化请求失败：服务器返回空响应")
    # 检查响应状态
    if resp.get("status") != "ok":
        error_msg = resp.get('msg', '未知错误')
        raise ConnectionError(f"初始化失败: {error_msg}")


class SimulationClient:
    """
    现在实现的是一个client会创建一个环境
    实际上支持一个client创建多个环境，env_id就可用于区分环境，待实现
    """
    def __init__(self, host: str, port: int, steps: int = 1, environment: str = "control",
                 log_save: bool = False, tac_view: bool = False):
        self.host = host
        self.port = port
        self.environment = environment
        self.client_id = str(uuid.uuid4())  # 用于环境唯一认证-日志保存
        if self.environment == "control":
            self.scenario = "testWzz"
            # 升降舵、副翼、方向舵、油门
            # 油门范围是[0,1]，其他范围是[-1,1]
            self.init_actions = [[0.5, 0.0, 0.0, 1.0]]

        elif self.environment == "dogfight":
            self.scenario = "test1v1"
            # 1V1环境 [副翼, 升降舵, 方向舵, 油门, 开火, 目标]
            self.init_actions = [
                [-0.05, 0.0, 0.0, 1.0, False, "5001"],
                [-0.05, 0.0, 0.0, 1.0, False, "1001"]
            ]

        elif self.environment == "multi3v3":
            self.scenario = "test1v1"
            # 1V1环境 [副翼, 升降舵, 方向舵, 油门, 开火, 目标]
            self.init_actions = [
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
            ]

        elif self.environment == "multi5v5":
            self.scenario = "test1v1"
            # 1V1环境 [副翼, 升降舵, 方向舵, 油门, 开火, 目标]
            self.init_actions = [
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
                [-0.05, 0.0, 0.0, 1.0, False, ""],
            ]
        else:
            # unsupported scenarios
            raise ValueError(f"不支持的环境类型: '{self.environment}'，支持的类型: 'control', 'dogfight', 'multi3v3', 'multi5v5'")
        self.init_payload = InitClass.get_parameter(self.environment)
        self.socket = None
        self.steps = steps
        self.log_save = log_save
        self.tac_view = tac_view
        self.logger = None
        self.viewer = None
        self.set_log_view()

    def set_log_view(self):
        if self.log_save:
            self.logger = TacViewLogger(env_id=self.client_id)
        if self.tac_view:
            self.viewer = TacView()

    def connection(self):
        """建立与C++服务器的连接

        单次通信仿真步长是16ms

        Returns:
            dict: 初始化响应数据

        Raises:
            ConnectionError: 连接失败或初始化失败时抛出
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            resp = self.send_request("init", self.init_payload)
            check_response(resp)  # 检查响应
            print(f"环境{self.environment}连接成功")
            return resp
        except ConnectionRefusedError:
            raise ConnectionError(
                f"无法连接到服务器 {self.host}:{self.port}，"
                f"请确认C++ Server是否正在运行"
            )
        except socket.timeout:
            raise ConnectionError(f"连接服务器超时")
        except Exception as e:
            traceback.print_exc()
            raise ConnectionError(f"连接过程中发生未预期错误: {e}")

    def get_environment_data(self, actions, env_id=0):
        if self.environment == "control":
            step_params = {
                "steps": self.steps,
                "actions": {
                    f"{env_id}": {"1001": actions[0]}
                }
            }
        elif self.environment == "dogfight":
            step_params = {
                "steps": self.steps,
                "actions": {
                    f"{env_id}": {"1001": actions[0], "5001": actions[1]}
                }
            }
        elif self.environment == "multi3v3":
            step_params = {
                "steps": self.steps,
                "actions": {
                    f"{env_id}": {"1001": actions[0], "1002": actions[1], "1003": actions[2],
                                  "5001": actions[3], "5002": actions[4], "5003": actions[5]}
                }
            }
        elif self.environment == "multi5v5":
            step_params = {
                "steps": self.steps,
                "actions": {
                    f"{env_id}": {"1001": actions[0], "1002": actions[1], "1003": actions[2],
                                  "1004": actions[3], "1005": actions[4],
                                  "5001": actions[5], "5002": actions[6], "5003": actions[7],
                                  "5004": actions[8], "5005": actions[9],
                                  }
                }
            }
        else:
            raise ValueError(f"不支持的环境类型: '{self.environment}'，"
                             f"支持的类型: 'control', 'dogfight'")
        resp = self.send_request("step", step_params)
        check_response(resp)
        if self.log_save:
            self.logger.add(resp["data"]["0"]["obs"])
        if self.tac_view:
            self.viewer.send_data_to_client(resp["data"]["0"]["obs"])
        return resp

    def reset(self, env_id=0):
        resp = self.send_request("reset", {"env_ids": [env_id], "scenario": self.scenario,
                                           "initial_state": self.init_payload["initial_state"]})
        check_response(resp)
        if self.log_save:
            self.logger = TacViewLogger(env_id=self.client_id)
        if self.tac_view:
            self.viewer.reconnect()
        return resp

    def close(self):
        try:
            resp = self.send_request("close", {"env_ids": [0]})
            check_response(resp)
        finally:
            if self.socket:
                self.socket.close()
            print("\n🔌 连接已关闭")

    def send_request(self, command, params):
        """封装好的发送函数"""
        req_id = f"{command}_{int(time.time() * 1000)}"
        payload = {
            "req_id": req_id, "cmd": command, "params": params
        }
        body_bytes = json.dumps(payload).encode('utf-8')
        header = struct.pack('<I', len(body_bytes))
        self.socket.sendall(header + body_bytes)

        header_recv = self.socket.recv(4)
        if not header_recv:
            raise ConnectionError("Connection closed")
        body_len = struct.unpack('<I', header_recv)[0]

        body_recv = b""
        while len(body_recv) < body_len:
            packet = self.socket.recv(body_len - len(body_recv))
            if not packet:
                break
            body_recv += packet

        return json.loads(body_recv)


if __name__ == "__main__":
    environment_name = "multi5v5"
    # environment_name = "multi3v3"
    # environment_name = "control"
    # environment_name = "dogfight"
    simulation = SimulationClient(host='127.0.0.1', port=8888, steps=20,
                                  environment=environment_name, log_save=True)
    r = simulation.connection()
    if environment_name == "control":
        actions = [[-0.2, 0.0, 0.0, 1.0]]
    elif environment_name == "dogfight":
        actions = [[-0.2, 0.0, 0.0, 1.0, False, "1001"], [-0.2, 0.0, 0.0, 1.0, False, "5001"]]
    elif environment_name == "multi3v3":
        actions = [[-0.05, 0.0, 0.0, 1.0, False, ""], [-0.05, 0.0, 0.0, 1.0, False, ""],
                   [-0.05, 0.0, 0.0, 1.0, False, ""], [-0.05, 0.0, 0.0, 1.0, False, ""],
                   [-0.05, 0.0, 0.0, 1.0, False, ""], [-0.05, 0.0, 0.0, 1.0, False, ""]]
    elif environment_name == "multi5v5":
        actions = [[-0.05, 0.0, 0.0, 1.0, False, ""], [-0.05, 0.0, 0.0, 1.0, False, ""],
                   [-0.05, 0.0, 0.0, 1.0, False, ""], [-0.05, 0.0, 0.0, 1.0, False, ""],
                   [-0.05, 0.0, 0.0, 1.0, False, ""], [-0.05, 0.0, 0.0, 1.0, False, ""],
                   [-0.05, 0.0, 0.0, 1.0, False, ""], [-0.05, 0.0, 0.0, 1.0, False, ""],
                   [-0.05, 0.0, 0.0, 1.0, False, ""], [-0.05, 0.0, 0.0, 1.0, False, ""]]
    else:
        raise ValueError(f"不支持的环境类型: '{environment_name}'，"
                         f"支持的类型: 'control', 'dogfight', 'multi3v3', 'multi5v5'")
    start_time = time.time()
    for x in range(1000):
        simulation.get_environment_data(actions, env_id=0)
    end_time = time.time()
    print("总耗时:", end_time - start_time, "秒")
    simulation.close()