<p align="center">
  📘 [中文说明](#-中文说明) ｜ 📗 [English Guide](#-english-guide)
</p>

---

# 🤖 中文说明

本项目是一个**简单、实用、高效**的多机器人控制平台，支持 **NRF24L01 + TTL 串口通信**，结合 **ArUco 标签实时识别与定位**，并通过**模型控制接口**实现机器人集体行为建模与路径规划。

## 🎬 功能演示（GIF 实时预览）

| 单机控制 | 多点定位 | 模型控制（方形轨迹） |
|:--------:|:--------:|:--------------------:|
| ![](gif/A_demonstration_of_a_single_machine.gif) | ![](gif/locate-multiple-points.gif) | ![](gif/model-control.gif) |

## 🧩 功能亮点

- 📍 多点定位：识别多个 ArUco 标签并实时获取其坐标。
- 🖼️ 可视化追踪：显示机器人路径、当前位置与目标点。
- 🧠 模型控制接口：可接入自定义模型（如路径规划、队形控制等）。
- 📡 NRF24L01 通信：通过 NRF24L01 模块无线发送角度+速度数据包。
- 🖱️ 两种控制模式：
  - 鼠标点击目标
  - 模型自动控制

## 🔌 模型接口规范

你可以通过 `set_model(func)` 方法传入一个控制模型，支持两种返回格式：

### 输入格式（机器人位置）：

```python
{
    0: (x0, y0),
    1: (x1, y1),
    ...
}
```

### 输出格式一：目标点

```python
{
    0: (tx0, ty0),
    1: (tx1, ty1),
    ...
}
```

### 输出格式二：路径点序列

```python
{
    "waypoints": {
        0: [(x1, y1), (x2, y2), ...],
        1: [(x1, y1), (x2, y2), ...]
    }
}
```

## 📦 串口数据包格式

| 字节 | 含义         |
|------|--------------|
| 0    | 机器人 ID    |
| 1    | 角度高 8 位   |
| 2    | 角度低 8 位   |
| 3    | 速度（0–255）|

说明：

- 角度范围 0–359°，以上方为 0，顺时针增加；
- 使用 `math.atan2` 计算方向，加偏移 +270°。

## 🛠️ 开发与运行

- Python 3.9+
- OpenCV 4.5+
- PySide6
- pyserial

安装依赖：

```bash
pip install opencv-python PySide6 pyserial
```

运行程序：

```bash
python main.py
```

## 📜 开源协议

本项目使用 [MIT License](LICENSE) 开源协议。

## 🙋 联系我们

欢迎通过 Issue、PR 提交建议或参与开发！

---

# 🇺🇸 English Guide

This project is a **simple, efficient, and flexible** multi-robot control GUI, featuring **NRF24L01 + TTL serial communication**, **real-time ArUco marker detection**, and a pluggable **model control interface** for path planning and coordination.

## 🎬 Demo Preview (GIF)

| Single Control | Multi-Point Localization | Model Control (Square) |
|:--------------:|:------------------------:|:----------------------:|
| ![](gif/A_demonstration_of_a_single_machine.gif) | ![](gif/locate-multiple-points.gif) | ![](gif/model-control.gif) |

## 🧩 Features

- 📍 Multi-point localization via ArUco markers
- 🖼️ Visual tracking and path display
- 🧠 Pluggable model control (e.g., square path, custom behavior)
- 📡 NRF24L01 wireless communication (angle + speed)
- 🖱️ Two modes:
  - Manual target via mouse click
  - Auto path control via model

## 🔌 Model Interface

```python
# Input:
{ 0: (x0, y0), 1: (x1, y1), ... }

# Output (1): target positions
{ 0: (tx0, ty0), 1: (tx1, ty1), ... }

# Output (2): waypoint lists
{ "waypoints": { 0: [(x1, y1), ...], 1: [(x1, y1), ...] } }
```

## 📦 Data Packet Format (serial)

| Byte | Description       |
|------|-------------------|
| 0    | Robot ID          |
| 1    | Angle (high byte) |
| 2    | Angle (low byte)  |
| 3    | Speed (0–255)     |

- Angle: 0° = up, clockwise increase;
- Calculated via `math.atan2` with +270° offset.

## 🛠️ Setup & Run

Requirements:

- Python 3.9+
- OpenCV 4.5+
- PySide6
- pyserial

Install:

```bash
pip install opencv-python PySide6 pyserial
```

Run:

```bash
python main.py
```

## 📜 License

Licensed under [MIT License](LICENSE).

## 🙋 Contact

Open issues or PRs to contribute or give feedback.
