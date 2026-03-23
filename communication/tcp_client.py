import socket
import json
import struct
import time
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
    def __init__(self, host: str, port: int, steps: int = 1, env_name: str = "control",
                 log_save: bool = False, tac_view: bool = False):
        self.host = host
        self.port = port
        self.env_name = env_name
        if env_name == "control":
            self.scenario = "testWzz"
            # 升降舵、副翼、方向舵、油门
            # 油门范围是[0,1]，其他范围是[-1,1]
            self.init_actions = [[0.5, 0.0, 0.0, 1.0]]
            self.target_ids = ["1001"]

        elif env_name == "dogfight":
            self.scenario = "test1v1"
            # 1V1环境 [副翼, 升降舵, 方向舵, 油门, 开火, 目标]
            self.init_actions = [
                [-0.05, 0.0, 0.0, 1.0, False, "5001"],
                [-0.05, 0.0, 0.0, 1.0, False, "1001"]
            ]
            self.target_ids = ["1001", "5001"]
        else:
            # unsupported scenarios
            raise ValueError(f"不支持的环境类型: '{env_name}'，支持的类型: 'control', 'dogfight'")
        self.init_payload = InitClass.get_parameter(env_name)
        self.socket = None
        self.steps = steps
        self.log_save = log_save
        self.tac_view = tac_view
        self.logger = None
        self.viewer = None
        self.set_log_view()

    def set_log_view(self):
        if self.log_save:
            self.logger = TacViewLogger()
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
            # self.init_payload["initial_state"]["1001"]["heading"] = 33
            resp = self.send_request("init", self.init_payload)
            check_response(resp)  # 检查响应
            resp = self.get_environment_data(self.init_actions)
            check_response(resp)  # 检查响应
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

    def get_environment_data(self, actions):
        if self.env_name == "control":
            step_params = {
                "steps": self.steps,
                "actions": {"0": {"1001": actions[0]}}
            }
        elif self.env_name == "dogfight":
            step_params = {
                "steps": self.steps,
                "actions": {"0": {"1001": actions[0], "5001": actions[1]}}
            }
        else:
            raise ValueError(f"不支持的环境类型: '{self.env_name}'，"
                             f"支持的类型: 'control', 'dogfight'")
        # print("step_params:", step_params)
        resp = self.send_request("step", step_params)
        # print(resp)
        check_response(resp)
        if self.log_save:
            self.logger.add(resp["data"]["0"]["obs"])
        if self.tac_view:
            self.viewer.send_data_to_client(resp["data"]["0"]["obs"])
        return resp

    def reset(self):
        # resp = self.send_request("reset", {"env_ids": [0]})
        # self.init_payload["initial_state"]["1001"]["heading"] = 111
        resp = self.send_request("reset", {"env_ids": [0], "scenario": self.scenario,
                                           "initial_state": self.init_payload["initial_state"]})
        # print("reset:", resp)
        check_response(resp)
        if self.log_save:
            self.logger = TacViewLogger()
        if self.tac_view:
            self.viewer.reconnect()
        resp = self.get_environment_data(self.init_actions)
        check_response(resp)  # 检查响应
        return resp

    def close(self):
        resp = self.send_request("close", {"env_ids": [0]})
        self.socket.close()
        print("\n🔌 连接已关闭")
        check_response(resp)

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
    env_name = "control"
    # env_name = "dogfight"
    simulation = SimulationClient(host='127.0.0.1', port=8888, steps=60,
                                  # env_name=env_name)
                                  # env_name=env_name, log_save=True, tac_view=True)
                                  env_name=env_name, log_save=True)
    observation = simulation.connection()
    # print(observation)
    if env_name == "control":
        actions = [[-0.2, 0.0, 0.0, 1.0]]
    elif env_name == "dogfight":
        actions = [[-0.2, 0.0, 0.0, 1.0, False, "1001"], [-0.2, 0.0, 0.0, 1.0, False, "5001"]]
    else:
        raise ValueError(f"不支持的环境类型: '{env_name}'，"
                         f"支持的类型: 'control', 'dogfight'")
    for i in range(20):
        observation = simulation.get_environment_data(actions)
        # time.sleep(1)
        # print(observation)
    observation = simulation.reset()
    print(observation)
    for i in range(20):
        observation = simulation.get_environment_data(actions)
        # time.sleep(1)
    # print(observation)
    simulation.close()
