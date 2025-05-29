import math
from typing import Dict, Tuple, Any

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
