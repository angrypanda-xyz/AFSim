import socket, os, time
from utils.tools import RAMathUtil


class TacView(object):
    """ TacView 实时数据传输处理类
    arguments:
        host: 服务器主机地址，默认本地地址:127.0.0.1 '
        port: 服务器端口，默认42674
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 42674):
        self.server_socket = None
        self.client_socket = None
        self.address = None
        self.host = host
        self.port = port
        self.setup_server()

    def setup_server(self):
        """ 设置服务器套接字并开始监听连接"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            print(f"Server listening on {self.host}:{self.port}")
            print(
                "IMPORTANT: Please open Tacview Advanced, click Record -> Real-time Telemetry, and input the IP address and port !")
            self.connect()
        except Exception as e:
            print(f"Setup error: {e}")
            self.cleanup()
            raise

    def send_data_to_client(self, observation):
        try:
            sim_time = observation["sim_time"]
            self.client_socket.send((f"#{sim_time:.2f}\n").encode())
            platforms = observation["platforms"]
            if not platforms:
                print("没有平台数据可记录")
                return
            for plane in platforms:
                data_line = RAMathUtil.plane_to_encode(plane)
                self.client_socket.send(data_line.encode())
        except Exception as e:
            print(f"Send error: {e}")
            self.reconnect()

    def connect(self):
        """ 等待客户端连接并进行握手 """
        try:
            print("Waiting for connection...")
            self.client_socket, self.address = self.server_socket.accept()
            print(f"Accepted connection from {self.address}")

            # 发送握手数据
            handshake_data = f"XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\n{socket.gethostname()}\n\x00"
            self.client_socket.send(handshake_data.encode())

            # 接收客户端响应
            data = self.client_socket.recv(1024)
            # print(f"Received data from {self.address}: {data.decode()}")

            # 发送头部数据
            current_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            header_data = (f"FileType=text/acmi/tacview\nFileVersion=2.2\n"
                           f"0,ReferenceTime={current_time}\n#0.00\n")
            self.client_socket.send(header_data.encode())
            print("Connection established")

        except Exception as e:
            print(f"Connection error: {e}")
            self.cleanup()
            raise

    def reconnect(self):
        """ 尝试重新连接客户端 """
        print("Attempting to reconnect...")
        self.cleanup()
        self.setup_server()

    def cleanup(self):
        try:
            if hasattr(self, 'client_socket') and self.client_socket:
                self.client_socket.close()
                self.client_socket = None
            if hasattr(self, 'server_socket') and self.server_socket:
                self.server_socket.close()
                self.server_socket = None
        except Exception as e:
            print(f"Cleanup error: {e}")

    def __del__(self):
        self.cleanup()


if __name__ == "__main__":
    viewer = TacView()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, 'F-16.acmi')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            viewer.client_socket.send(line.encode())
            # 每 0.01s 发送一行数据
            time.sleep(0.01)
