class InitClass:
    @staticmethod
    def get_parameter(env_name, env_num=1):
        if env_name == "control":
            init_json = {
                "count": env_num,  # [整数] 并行生成的环境数量（测试填 1，RL 训练时可以填多个）
                "scenario": "testWzz",  # / [字符串] 调用的纯净模板名（对应 test1v1.txt）
                "initial_state": {  # [字典] 动态注入的初始状态，Key 是飞机的 ID
                    "1001": {  # 红方飞机 ID
                        "type": "F-16",  # [字符串] 模板里定义的平台类型，服务端支持jian20和F-16
                        "side": "red",  # [字符串] 阵营（"red" 或 "blue"）

                        "lat": 35.6666,  # [浮点数] 纬度 (度)
                        "lon": 117.0,  # [浮点数] 经度 (度)
                        "alt": 10000.0,  # [浮点数] 高度 (纯纯的公制：米！千万别再换算英尺了)

                        # ---  飞行姿态与动能 ---
                        "speed": 350.0,  # [浮点数] 初始速度 (米/秒)
                        "heading": 90.0,  # [浮点数] 偏航角/航向 (度，0是正北，90是正东)
                        "pitch": 0.0,  # [浮点数] 俯仰角 (度，正数抬头，负数低头)
                        "roll": 0.0,  # [浮点数] 滚转角 (度)
                    }
                }
            }
        elif env_name == "dogfight":
            init_json = {
                "count": env_num,  # [整数] 并行生成的环境数量（测试填 1，RL 训练时可以填多个）
                "scenario": "test1v1",  # / [字符串] 调用的纯净模板名（对应 test1v1.txt）
                "initial_state": {  # [字典] 动态注入的初始状态，Key 是飞机的 ID
                    "1001": {  # 红方飞机 ID
                        "type": "F-16",  # [字符串] 模板里定义的平台类型
                        "side": "Red",  # [字符串] 阵营（"Red" 或 "Blue"）

                        "lat": 35.6666,  # [浮点数] 纬度 (度)
                        "lon": 117.0,  # [浮点数] 经度 (度)
                        "alt": 10000.0,  # [浮点数] 高度 (纯纯的公制：米！千万别再换算英尺了)

                        # ---  飞行姿态与动能 ---
                        "speed": 350.0,  # [浮点数] 初始速度 (米/秒)
                        "heading": 90.0,  # [浮点数] 偏航角/航向 (度，0是正北，90是正东)
                        "pitch": 0.0,  # [浮点数] 俯仰角 (度，正数抬头，负数低头)
                        "roll": 0.0,  # [浮点数] 滚转角 (度)

                        # ---  武器挂载 ---
                        "weapons": {  # [字典] 武器插槽与数量
                            "AIM-9": 2  # Key 必须和 txt 里的 weapon 插槽名一致，Value 是挂弹量
                        }
                    },
                    "5001": {  # 蓝方飞机 ID
                        "type": "F-16",
                        "side": "Blue",
                        "lat": 35.6666,
                        "lon": 117.6,  # 经度拉开，制造初始距离
                        "alt": 10000.0,
                        "speed": 300.0,
                        "heading": 270.0,  # 270度是正西，和红方形成迎头对飞
                        "pitch": 0.0,
                        "roll": 0.0,
                        "weapons": {
                            "AIM-9": 0  # 靶机不挂弹
                        }
                    }
                }
            }
        elif env_name == "multi3v3":
            init_json = {
                "count": 1,
                "scenario": "test1v1",
                "initial_state": {
                    # --- 🔴 红方中队 (1001, 1002, 1003) ---
                    "1001": {"type": "jian20", "side": "Red", "lat": 35.0, "lon": 117.0, "alt": 10000.0,
                             "heading": 90.0, "speed": 300.0, "weapons": {"aim9x": 2}},
                    "1002": {"type": "jian20", "side": "Red", "lat": 35.05, "lon": 116.95, "alt": 10000.0,
                             "heading": 90.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 左翼僚机
                    "1003": {"type": "jian20", "side": "Red", "lat": 34.95, "lon": 116.95, "alt": 10000.0,
                             "heading": 90.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 右翼僚机
                    # --- 🔵 蓝方中队 (5001, 5002, 5003) ---
                    "5001": {"type": "F-16", "side": "Blue", "lat": 35.0, "lon": 117.6, "alt": 10000.0,
                             "heading": 270.0, "speed": 300.0, "weapons": {"aim9x": 2}},
                    "5002": {"type": "F-16", "side": "Blue", "lat": 35.05, "lon": 117.65, "alt": 10000.0,
                             "heading": 270.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 左翼僚机
                    "5003": {"type": "F-16", "side": "Blue", "lat": 34.95, "lon": 117.65, "alt": 10000.0,
                             "heading": 270.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 右翼僚机
                }
            }
        elif env_name == "multi5v5":
            init_json = {
                "count": 1,
                "scenario": "test1v1",
                "initial_state": {
                    # --- 🔴 红方中队 (1001, 1002, 1003) ---
                    "1001": {"type": "jian20", "side": "Red", "lat": 35.0, "lon": 117.0, "alt": 10000.0,
                             "heading": 90.0, "speed": 300.0, "weapons": {"aim9x": 2}},
                    "1002": {"type": "jian20", "side": "Red", "lat": 35.05, "lon": 116.95, "alt": 10000.0,
                             "heading": 90.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 左翼僚机
                    "1003": {"type": "jian20", "side": "Red", "lat": 34.95, "lon": 116.95, "alt": 10000.0,
                             "heading": 90.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 右翼僚机
                    "1004": {"type": "jian20", "side": "Red", "lat": 35.05, "lon": 117.05, "alt": 10000.0,
                             "heading": 90.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 左翼僚机
                    "1005": {"type": "jian20", "side": "Red", "lat": 34.95, "lon": 117.05, "alt": 10000.0,
                             "heading": 90.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 右翼僚机

                    # --- 🔵 蓝方中队 (5001, 5002, 5003) ---
                    "5001": {"type": "F-16", "side": "Blue", "lat": 35.0, "lon": 117.6, "alt": 10000.0,
                             "heading": 270.0, "speed": 300.0, "weapons": {"aim9x": 2}},
                    "5002": {"type": "F-16", "side": "Blue", "lat": 35.05, "lon": 117.65, "alt": 10000.0,
                             "heading": 270.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 左翼僚机
                    "5003": {"type": "F-16", "side": "Blue", "lat": 34.95, "lon": 117.65, "alt": 10000.0,
                             "heading": 270.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 右翼僚机
                    "5004": {"type": "F-16", "side": "Blue", "lat": 35.05, "lon": 117.55, "alt": 10000.0,
                             "heading": 270.0, "speed": 300.0, "weapons": {"aim9x": 2}},  # 左翼僚机
                    "5005": {"type": "F-16", "side": "Blue", "lat": 34.95, "lon": 117.55, "alt": 10000.0,
                             "heading": 270.0, "speed": 300.0, "weapons": {"aim9x": 2}}  # 右翼僚机
                }
            }
        else:
            init_json = None
        return init_json


if __name__ == "__main__":
    result = InitClass.get_parameter("control")
    print(result)
