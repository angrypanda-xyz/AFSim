# SimulationClient README

## 项目概述

这是一个与C++仿真服务器进行通信的Python客户端。它通过TCP socket与服务器建立连接，发送控制指令并接收仿真数据，主要用于飞机仿真环境的控制和数据采集。

## 功能特性

- 建立与C++服务器的TCP连接
- 初始化仿真场景
- 发送控制指令（升降舵、副翼、方向舵、油门）
- 接收环境状态数据
- 支持多步仿真推进
- 自动等待目标飞机上线
- 仿真重置和关闭功能

## 安装要求

- Python 3.6+
- 无需额外安装第三方库（仅使用Python标准库）

## 快速开始

### 1. 启动C++服务器

确保C++服务器已在本地端口8888上运行。

### 2. 运行示例代码

```python
from simulation_client import SimulationClient

# 创建客户端实例
# steps=60 表示每一步仿真是1/60秒
simulation = SimulationClient(host='127.0.0.1', port=8888, steps=60)

# 连接并初始化场景
observation = simulation.connection(scenario="testWzz")

# 运行100步仿真
for i in range(100):
    # 发送控制指令 [升降舵, 副翼, 方向舵, 油门]
    observation = simulation.get_environment_data([[0.5, 0.0, 0.0, 1.0]])
    print(json.dumps(observation))

# 重置仿真环境
simulation.reset()

# 再次运行一步
observation = simulation.get_environment_data([[0.5, 0.0, 0.0, 1.0]])

# 关闭连接
simulation.close()
```

## 类和方法说明

### SimulationClient 类

#### 构造函数

```python
SimulationClient(host: str, port: int, steps: int = 1)
```

- `host`: 服务器IP地址
- `port`: 服务器端口号
- `steps`: 每一步仿真的时间步长（单位：1/60秒）

#### 主要方法

##### connection(scenario)

建立与服务器的连接并初始化场景。

- `scenario`: 场景名称，如 "testWzz"
- 返回: 服务器响应数据

##### get_environment_data(actions)

发送控制指令并获取环境数据。

- `actions`: 控制指令列表，格式为 `[[升降舵, 副翼, 方向舵, 油门]]`
  - 升降舵、副翼、方向舵范围: [-1, 1]
  - 油门范围: [0, 1]
- 返回: 包含环境状态数据的JSON对象

##### reset()

重置仿真环境到初始状态。

##### close()

关闭与服务器的连接。

##### wait_for_platform_ready(target_id, timeout=10)

轮询等待指定飞机上线。

- `target_id`: 目标飞机ID
- `timeout`: 超时时间（秒）
- 返回: (是否成功, 观测数据)

## 通信协议

### 请求格式

客户端发送的消息包含4字节小端序长度头 + JSON格式的请求体。

请求体格式：

```json
{
    "req_id": "请求ID_时间戳",
    "cmd": "命令名称",
    "params": {
        // 命令参数
    }
}
```

### 支持的命令

- `init`: 初始化场景
- `step`: 执行仿真步骤
- `reset`: 重置环境
- `close`: 关闭连接

### 控制指令说明

控制指令为4个浮点数组成的列表：

1. 升降舵 (elevator): [-1, 1]
2. 副翼 (aileron): [-1, 1]  
3. 方向舵 (rudder): [-1, 1]
4. 油门 (throttle): [0, 1]

## 场景配置

当前支持两种场景：

- `testWzz`: 测试场景，目标飞机ID为"1001"
- 其他场景: 目标飞机列表为空

## 错误处理

- 连接失败时会打印错误信息并退出
- 初始化失败时会显示具体失败原因
- 网络异常会捕获并打印堆栈信息

## 注意事项

1. 确保C++服务器已启动并在指定端口监听
2. 默认使用本地回环地址(127.0.0.1)和端口8888
3. 控制指令值必须在指定范围内
4. 等待飞机上线有10秒超时限制
5. 通信使用小端序字节序

## 示例输出

成功连接时的输出示例：

```
🕵️‍♂️ 正在轮询等待飞机 [1001] 上线 (超时: 10s)...
✅ 成功捕获目标！飞机 [1001] 已就绪。
```

