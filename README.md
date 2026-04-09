# AFSim 飞行仿真算法项目

## 项目简介

AFSim是一个基于强化学习的飞行仿真算法项目，专注于飞行器控制、空战对抗等场景的智能体训练与仿真。项目实现了多种环境下的智能控制算法，并提供了完整的训练、可视化和通信模块。

本项目需要配合蓉奥科技公司开发的“基于AFSim的仿真训练平台”使用（备注：该平台免费提供给空战爱好者使用，解压后运行RASimManager.exe即可）。

该平台在百度网盘下载链接为: https://pan.baidu.com/s/1x8I2QVLVP2i3juIdZPnc3w?pwd=brey 提取码: brey


## 项目结构

```json
AFSim/
├── communication/ # 通信模块
│ └── tcp_client.py # TCP客户端实现
├── environments/ # 仿真环境
│ ├── aircraf_control/ # 飞行器控制环境
│ ├── dogfight_1v1/ # 1v1空战对抗环境
│ └── point_tracking/ # 点跟踪环境
├── training/ # 训练模块
│ ├── aircraf_control_traing/ # 飞行器控制训练
│ ├── dogfight_1v1_traing/ # 空战对抗训练
│ └── point_tracking_traing/ # 点跟踪训练
├── utils/ # 工具模块
│ └── tools.py # 通用工具函数
└── visualization/ # 可视化模块
│ └── logger_handler.py # 日志处理
│ └── tacview_handler.py # Tacview可视化处理
```

## 环境说明

### 1. 飞行器控制环境 (aircraf_control)

- 基础的飞行器控制仿真环境
- 包含飞行器动力学模型
- 支持姿态控制、轨迹跟踪等任务

### 2. 1v1空战对抗环境 (dogfight_1v1)

- 双机空战对抗仿真
- 包含基本的空战机动决策
- 支持红蓝双方对抗训练

### 3. 点跟踪环境 (point_tracking)

- 简单的位置跟踪任务
- 适合算法验证和调试
- 提供基础的控制基准

## 训练模块

### 飞行器控制训练

- 实现飞行器姿态控制算法
- 支持PID、强化学习等多种控制方法
- 提供训练监控和评估指标

### 空战对抗训练

- 多智能体强化学习训练框架
- 支持自博弈训练
- 包含胜负判定和奖励机制

### 点跟踪训练

- 基础控制算法训练
- 适合快速验证算法效果
- 提供详细的训练日志

## 通信模块

`tcp_client.py` 提供了TCP通信接口，支持：

- 与外部仿真器（如Tacview）的数据交互
- 分布式训练时的数据传输
- 实时状态同步

## 可视化模块

### Logger Handler

- 训练日志记录和管理
- 支持多种日志格式输出
- 提供训练曲线可视化

### Tacview Handler

- 与Tacview仿真可视化软件的接口
- 将仿真数据转换为Tacview格式
- 支持飞行轨迹回放和分析

## 工具模块

`tools.py` 包含通用工具函数：

- 数据预处理
- 数学计算辅助函数
- 配置文件解析
- 模型保存和加载

## 快速开始

### 环境要求

```bash
Python >= 3.7
numpy
pytorch >= 1.8
gym >= 0.18.0
```
面向六自由度飞行器模型和AFSim仿真服务，实现的强化学习训练客户端
![仿真端](https://github.com/user-attachments/assets/27cc3012-1115-4193-b487-ab6c06e4d43e)


<img width="1528" height="6132" alt="交互流程图" src="https://github.com/user-attachments/assets/0cdc9e25-8270-4fc7-bfff-802a783ce5c4" />

<img width="1450" height="5739" alt="训练流程" src="https://github.com/user-attachments/assets/e0953233-f141-4602-89f0-b527479d69e5" />
