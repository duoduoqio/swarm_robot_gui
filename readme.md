# 🤖 多机器人定位与控制系统

本项目是一个**简单、实用、高效**的多机器人控制平台，支持 **NRF24L01 + TTL 串口通信**，结合 **ArUco 标签实时识别与定位**，并通过**模型控制接口**实现机器人集体行为建模与路径规划。

---

## 🎬 功能演示（GIF）

| 单机控制 | 多点定位 | 模型控制（方形轨迹） |
|:--------:|:--------:|:--------------------:|
| ![](gif/A_demonstration_of_a_single_machine.gif) | ![](gif/locate-multiple-points.gif) | ![](gif/model-control.gif) |

---

## 🧩 功能亮点

- 📍 多点定位：识别多个 ArUco 标签并实时获取其坐标。
- 🖼️ 可视化追踪：显示机器人路径、当前位置与目标点。
- 🧠 模型控制接口：可接入自定义模型（如路径规划、队形控制等）。
- 📡 NRF24L01 通信：通过 NRF24L01 模块无线发送角度+速度数据包。
- 🖱️ 两种控制模式：
  - 鼠标点击目标
  - 模型自动控制

---

## 🔌 模型接口规范

```python
# 输入格式：{机器人ID: (x, y)}
{
    0: (x0, y0),
    1: (x1, y1),
    ...
}

# 输出格式一：直接目标
{
    0: (tx0, ty0),
    1: (tx1, ty1),
    ...
}

# 输出格式二：路径点序列
{
    "waypoints": {
        0: [(x1, y1), (x2, y2)],
        1: [(x3, y3), (x4, y4)]
    }
}
```

---

## 📦 串口数据包格式

| 字节 | 含义         |
|------|--------------|
| 0    | 机器人 ID    |
| 1    | 角度高 8 位   |
| 2    | 角度低 8 位   |
| 3    | 速度（0–255）|

说明：

- 角度范围 0~359°，正上方为 0°，顺时针方向；
- 使用 `math.atan2()` 计算方向，并加偏移 +270°。

---

## 🛠️ 开发与运行

依赖环境：

- Python 3.9+
- OpenCV 4.5+
- PySide6
- pyserial

安装依赖：

```bash
pip install opencv-python PySide6 pyserial
```

运行：

```bash
python main.py
```

---

## 📜 开源协议

本项目使用 [MIT License](LICENSE) 协议，欢迎使用与二次开发。

---

<details>
<summary>🇺🇸 Click to view English version</summary>

## 🤖 Multi-Robot Localization & Control System

This project is a **simple, practical and efficient** multi-robot control platform, using **NRF24L01 + TTL communication**, real-time **ArUco tag detection**, and a **pluggable model interface** for behavior generation and path planning.

---

### 🎬 Demo Preview (GIF)

| Single Robot | Multi-Point Tracking | Square Model |
|:------------:|:--------------------:|:------------:|
| ![](gif/A_demonstration_of_a_single_machine.gif) | ![](gif/locate-multiple-points.gif) | ![](gif/model-control.gif) |

---

### 🧩 Features

- 📍 Multi-point real-time localization with ArUco
- 🖼️ Visualization of path, targets and positions
- 🧠 Plug-in model interface for path/behavior control
- 📡 NRF24L01 wireless control via 4-byte packets
- 🖱️ Two modes:
  - Manual target mode (via mouse)
  - Model-based control mode

---

### 🔌 Model Interface (Python)

```python
# Input: robot positions
{ 0: (x0, y0), 1: (x1, y1), ... }

# Output A: direct target positions
{ 0: (tx0, ty0), 1: (tx1, ty1), ... }

# Output B: waypoint list per robot
{ "waypoints": { 0: [(x1, y1), (x2, y2)], 1: [(x3, y3), (x4, y4)] } }
```

---

### 📦 Data Packet Format

| Byte | Meaning         |
|------|-----------------|
| 0    | Robot ID        |
| 1    | Angle high byte |
| 2    | Angle low byte  |
| 3    | Speed (0–255)   |

Angle is calculated as:

```python
angle = int((atan2(ty - ry, tx - rx) * 180 / pi + 360 + 270) % 360)
```

---

### 🛠️ Setup

```bash
pip install opencv-python PySide6 pyserial
python main.py
```

---

### 📜 License

MIT License

</details>
