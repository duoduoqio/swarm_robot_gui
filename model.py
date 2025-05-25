import math
from typing import Dict, Tuple, Any, List
import time

from env import Environment  # 导入环境类
from swarm_robot import SwarmRobot  # 导入机器人类

def square_model(pos_dict: Dict[int, Tuple[float, float]]) -> Any:
    """
    为每个机器人生成一组密集的 waypoints，使其从当前位置出发，
    顺时针绕当前点画一个边长 0.1m 的正方形。

    输入:
      pos_dict: { robot_id: (wx, wy), ... }

    输出:
      {
        'waypoints': {
          robot_id: [(x0,y0), (x1,y1), ..., (xN,yN)],
          ...
        }
      }
    """
    side = 0.1           # 边长 0.1m
    pts_per_edge = 20    # 每条边分成 20 段，产生 21 个采样点（包含端点）
    waypoints: Dict[int, list[Tuple[float, float]]] = {}

    print(f"pos_dict: {pos_dict}")

    for rid, (cx, cy) in pos_dict.items():
        pts: list[Tuple[float, float]] = []

        # 定义正方形四个角相对位置
        corners = [
            (cx,            cy),
            (cx + side,     cy),
            (cx + side, cy + side),
            (cx,        cy + side),
            (cx,            cy)  # 回到起点
        ]

        # 沿每条边线性插值
        for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
            for i in range(pts_per_edge):
                t = i / pts_per_edge
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                pts.append((x, y))
        # 最后再加回起点
        pts.append(corners[-1])

        waypoints[rid] = pts

    return {'waypoints': waypoints}


# 每个机器人的路径缓存与当前点索引
_dense_paths: Dict[int, List[Tuple[float, float]]] = {}
_path_index: Dict[int, int] = {}

def square_dense_model(pos_dict: Dict[int, Tuple[float, float]]) -> Dict[int, Tuple[float, float]]:
    """
    每个机器人沿稠密正方形轨迹连续移动，到达一个点就进入下一个。
    """
    side = 0.2         # 正方形边长
    pts_per_edge = 20  # 每边插值点数（越大越平滑）
    tolerance = 0.01   # 判定“到达目标点”的距离阈值

    result: Dict[int, Tuple[float, float]] = {}

    for rid, (cx, cy) in pos_dict.items():
        # 若第一次进入，为其生成正方形路径
        if rid not in _dense_paths:
            corners = [
                (cx, cy),
                (cx + side, cy),
                (cx + side, cy + side),
                (cx, cy + side),
                (cx, cy)
            ]
            pts: List[Tuple[float, float]] = []
            for (x0, y0), (x1, y1) in zip(corners, corners[1:]):
                for i in range(pts_per_edge):
                    t = i / pts_per_edge
                    x = x0 + (x1 - x0) * t
                    y = y0 + (y1 - y0) * t
                    pts.append((x, y))
            pts.append(corners[-1])
            _dense_paths[rid] = pts
            _path_index[rid] = 0

        path = _dense_paths[rid]
        idx = _path_index[rid]
        tx, ty = path[idx]

        # 如果到达当前目标点，切换到下一个
        if math.hypot(tx - cx, ty - cy) < tolerance:
            idx = (idx + 1) % len(path)
            _path_index[rid] = idx
            tx, ty = path[idx]

        result[rid] = (tx, ty)

    print(f"Dense square targets: {result}")
    return result



def sph_model(pos_dict: Dict[int, Tuple[float, float]], env:Environment, swarm_robot:SwarmRobot) -> Dict[int, Tuple[float, float]]:
      
    robotlist = swarm_robot.robot_list

    #根据pos_dict中的坐标更新机器人的位置
    robotlist = update_robot_position(robotlist, pos_dict)


    next_state_robotlist = swarm_robot.get_waypoints(robotlist)

    
    

def update_robot_position(robotlist, pos_dict):
    """
    更新机器人的位置
    """

    # TODO:需要坐标映射
    for rid, (x, y) in pos_dict.items():
        robotlist[rid].position = (x, y)
    return robotlist