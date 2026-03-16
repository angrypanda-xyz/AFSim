# TacView实时遥测数据传输类 README

## 项目概述

这是一个用于与Tacview软件建立实时遥测数据连接的Python类。它通过TCP socket模拟Tacview实时遥测服务器，将ACMI格式的飞行数据实时传输到Tacview客户端进行可视化展示。

## 功能特性

- 建立TCP服务器监听Tacview客户端连接
- 自动处理Tacview握手协议
- 实时传输ACMI格式的遥测数据
- 支持断线重连机制
- 自动清理网络资源

## 安装要求

- Python 3.6+
- 无需额外安装第三方库（仅使用Python标准库）

## Tacview配置说明

### Tacview客户端设置

1. 打开Tacview Advanced版本
2. 点击 **Record** → **Real-time Telemetry**
3. 在弹出的窗口中输入本机IP地址（默认：127.0.0.1）和端口（默认：42674）
4. 点击 **OK** 开始等待数据连接

## 快速开始

### 1. 基础用法

```python
from tacview_handler import TacView
import time

# 创建TacView服务器实例（使用默认地址和端口）
tacview = TacView()  # 默认 host='127.0.0.1', port=42674

# 或者自定义地址和端口
# tacview = TacView(host='0.0.0.0', port=55555)

# 发送单条ACMI数据
tacview.send_data_to_client("0,1,2,3,4,5\n".encode())
```

### 2. 从文件发送ACMI数据

```python
from tacview_handler import TacView
import os
import time

tacview = TacView()

# 读取并发送ACMI文件内容
acmi_file_path = 'your_flight_data.acmi'
with open(acmi_file_path, 'r') as f:
    for line in f:
        tacview.send_data_to_client(line.encode())
        time.sleep(0.1)  # 控制发送速率
```

## 类和方法说明

### TacView 类

#### 构造函数

```python
TacView(host: str = '127.0.0.1', port: int = 42674)
```

- `host`: 服务器监听地址，默认本地回环地址
- `port`: 服务器监听端口，默认Tacview标准端口42674

#### 主要方法

##### setup_server()

初始化并配置服务器套接字，开始监听客户端连接。

- 自动调用`connect()`方法等待客户端连接
- 设置`SO_REUSEADDR`选项避免地址占用问题

##### connect()

等待Tacview客户端连接并进行协议握手：

1. 接受客户端连接
2. 发送握手协议数据
3. 接收客户端响应
4. 发送ACMI文件头（包含参考时间）
5. 确认连接建立

##### send_data_to_client(data)

向已连接的Tacview客户端发送数据。

- `data`: 要发送的二进制数据（bytes类型）
- 发送失败时自动尝试重连

##### reconnect()

当连接断开时尝试重新建立连接：

1. 清理现有连接
2. 重新调用`setup_server()`等待新连接

##### cleanup()

清理网络资源：

- 关闭客户端socket
- 关闭服务器socket
- 重置socket对象

## ACMI数据格式说明

### 数据格式要求

本类期望接收**不含文件头**的纯ACMI数据部分。文件头会在连接建立时自动发送。

正确的数据格式示例：

```
#0.01
0,1,2,3,4,5
#0.02
0,6,7,8,9,10
```

### ACMI文件预处理

如果您的ACMI文件包含完整的头部信息，需要先去除头部：

```python
def process_acmi_file(file_path):
    """处理ACMI文件，去除头部信息"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 跳过文件头（前三行）
    return lines[3:]  # 返回纯数据部分
```

## 通信协议详解

### 握手流程

1. **服务器发送**：`XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\n{hostname}\n\x00`
2. **客户端响应**：确认消息
3. **服务器发送**：ACMI文件头（包含参考时间）
4. **数据传输**：持续的ACMI数据流

### 时间同步

参考时间使用UTC格式：

```
0,ReferenceTime=2024-01-15T08:30:45Z
```

## 错误处理

### 常见异常及处理

- **连接失败**：自动进入等待重连状态
- **发送失败**：自动尝试重连后重发
- **端口占用**：检查是否有其他程序占用端口

### 重连机制

```python
try:
    self.client_socket.send(data)
except Exception:
    self.reconnect()  # 自动重连
    self.client_socket.send(data)  # 重试发送
```

## 注意事项

1. **Tacview版本要求**：需要使用Tacview Advanced版本，免费版不支持实时遥测功能
2. **防火墙设置**：确保防火墙允许Python程序监听指定端口
3. **连接顺序**：必须先启动本程序，再在Tacview中连接
4. **数据格式**：确保发送的数据符合ACMI格式规范
5. **时间控制**：通过`time.sleep()`控制数据发送速率，避免网络拥塞
