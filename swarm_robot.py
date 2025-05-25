import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from env import Environment
from SPH_model.model import Robot, BaseSPH
from scipy.spatial import cKDTree
from typing import List

class SwarmRobot:
    def __init__(self, env: Environment, model="BaseSPH", dt=0.5, seed=0):
        np.random.seed(seed=seed)
        self.swarm_size = env.swarm_size
        self.mode_name = model
        
        self.env = env
        self.rn, self.cn = env.rn_scaled, env.cn_scaled
        self.robot_list = []
        # 参数设置
        self.r_body = 0.5
        self.r_sense = env.r_sense
        self.max_velocity = 0.1
        self.dt = dt

        # 初始化时直接调用添加方法
        self.spatial_grid = SpatialGrid(cell_size = env.r_sense * 1.5)  # 网格尺寸建议为感知半径的1.5倍

        self.init_add_robot(env.swarm_size)
        # self.add_robots(env.swarm_size)
        
        self.boundary_mode = self.env.boundary_mode
        # 虚粒子坐标
        if self.boundary_mode == "virtual_particles":
            self.virtual_boundary_pos = env.map_index_to_robot_pos_scaled_for_virtual_partical()
            self.add_virtual_boundary()
        # self.init_robot_position()
        # self.init_robot_velocity()
      
        
        if model == "BaseSPH":
            self.model = BaseSPH(self.robot_list, env, self.r_sense,self.max_velocity,self.dt )


    def init_add_robot(self, N):
        '''添加N个机器人，初始坐标和速度都为0'''
        for i in range(N):
            # 生成有效位置
            existing_positions = [robot.position for robot in self.robot_list]
            position = np.array([0, 0])
            # 生成初始速度
            velocity = np.array([0, 0])
            # 创建机器人实例（ID连续）
            new_robot = Robot(i, position.copy(), velocity.copy())
            self.robot_list.append(new_robot)

            self.spatial_grid.update_robot(new_robot.R_id, position)
        
    def add_robots(self, N):
        """添加N个机器人"""
        current_count = len(self.robot_list)
        for i in range(N):
            # 生成有效位置
            existing_positions = [robot.position for robot in self.robot_list]
            position = self._generate_valid_position(existing_positions)
            # 生成初始速度
            velocity = self._generate_initial_velocity()
            # 创建机器人实例（ID连续）
            new_robot = Robot(current_count + i, position.copy(), velocity.copy())
            self.robot_list.append(new_robot)

            self.spatial_grid.update_robot(new_robot.R_id, position)  # 新增

    def add_virtual_boundary(self):
        current_count = len(self.robot_list)
        for i in range(len(self.virtual_boundary_pos)):
            # 生成有效位置
            position = np.array(self.virtual_boundary_pos[i])
            # 生成初始速度
            velocity = np.array([0, 0])
            # 创建机器人实例（ID连续）
            new_robot = Robot(current_count + i, position.copy(), velocity.copy(), True)
            self.robot_list.append(new_robot)

            self.spatial_grid.update_robot(new_robot.R_id, position)  # 新增

    def remove_robots(self, N, remove_virtual=False):
        """安全删除机器人（支持删除普通/虚粒子）"""
        if N <= 0 or len(self.robot_list) == 0:
            return
        
        # 确定要删除的机器人范围
        if N >= len(self.robot_list):
            targets = self.robot_list.copy()
        else:
            targets = self.robot_list[-N:]
        
        # 根据类型过滤
        if not remove_virtual:
            targets = [r for r in targets if not r.virtual_boundary]
            N = len(targets)
        
        # 从网格中移除
        for robot in targets:
            self.spatial_grid.remove_robot(robot.R_id)
        
        # 从主列表中移除
        if N >= len(self.robot_list):
            self.robot_list.clear()
        else:
            self.robot_list = self.robot_list[:-N]

    def _generate_valid_position(self, existing_positions):
        """生成不与现有位置冲突的新位置"""
        rn, cn = self.rn, self.cn
        while True:
            ratio = 1.2
            position = np.random.rand(2) * np.array([rn/ratio, cn/ratio]) + \
                     np.array([rn/2-rn/(2*ratio), cn/2-cn/(2*ratio)])
            if all(np.linalg.norm(position - p) >= 3*self.r_body for p in existing_positions):
                return position

    def _generate_initial_velocity(self):
        """生成初始速度"""
        theta = np.random.uniform(0, 2*np.pi)
        return np.array([self.max_velocity/4*np.cos(theta),
                        self.max_velocity/4*np.sin(theta)])

    def move_robots(self):
        # 更新所有机器人的位置
        for robot_i in self.robot_list:
            # 更新网格位置
            if not robot_i.virtual_boundary:  # 假设虚粒子不移动
                robot_i.position = robot_i.position + self.dt * robot_i.velocity
                self.spatial_grid.update_robot(robot_i.R_id, robot_i.position)
                robot_i.update(robot_i.position, robot_i.velocity)


    def calculate_neighbors(self): # 这里可以更加细化，比如要求两个机器人之间连线也处于同一label
        """优化后的邻居搜索（加速10-100倍）"""
        all_neighbors = [[] for _ in range(len(self.robot_list))]
        
        for i, robot in enumerate(self.robot_list):
            if robot.virtual_boundary:
                continue  # 虚粒子不需要邻居
            
            # Step 1: 通过网格快速获取候选邻居
            candidate_ids = self.spatial_grid.query_neighbors(
                robot.position, self.r_sense
            )
            
            # Step 2: 精确距离过滤
            valid_neighbors = []
            robot_all_neighbors = []
            for j in candidate_ids:
                other = self.robot_list[j]
                
                # 基础过滤条件
                if (other.R_id == robot.R_id or 
                    np.linalg.norm(robot.position - other.position) > self.r_sense or # 通信范围
                    (robot.label <= 0 and other.virtual_boundary)): # 虚粒子只对内部粒子起作用
                    continue
                
                robot_all_neighbors.append(other)

                # 新增区域连通性检查
                if robot.label > 0:
                    # 条件1：邻居label必须为0或当前label
                    if not (other.label == 0 or other.label == robot.label):
                        continue
                
                valid_neighbors.append(other)
            
            robot.neighbors = valid_neighbors
            robot.neighbors_all = robot_all_neighbors
    
    def in_target_area(self):
        for robot in self.robot_list:
            robot.label = self.env.is_in_target_area(robot.position)    
    

    def run(self):
        self.calculate_neighbors()
        self.robot_list = self.model.run(self.robot_list)
        self.move_robots()
        self.in_target_area()

  
        
    def get_waypoints(self, robotlist):

        self.robot_list = robotlist
        self.calculate_neighbors()
        self.robot_list = self.model.run(self.robot_list)
        self.in_target_area()

        return self.robot_list

from collections import defaultdict

class SpatialGrid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = defaultdict(list)  # {grid_key: [robot_indices]}
        self.robot_grid_map = {}       # {robot_id: grid_key}

    def update_robot(self, robot_id, position):
        """更新机器人所在网格"""
        # 删除旧记录
        if robot_id in self.robot_grid_map:
            old_key = self.robot_grid_map[robot_id]
            self.grid[old_key].remove(robot_id)
            if not self.grid[old_key]:
                del self.grid[old_key]
        
        # 计算新网格键
        new_key = tuple((position // self.cell_size).astype(int))
        
        # 更新记录
        self.grid[new_key].append(robot_id)
        self.robot_grid_map[robot_id] = new_key

    def remove_robot(self, robot_id):
        """从网格中删除指定ID的机器人"""
        if robot_id in self.robot_grid_map:
            grid_key = self.robot_grid_map[robot_id]
            try:
                self.grid[grid_key].remove(robot_id)
                # 清理空网格
                if not self.grid[grid_key]:
                    del self.grid[grid_key]
            except ValueError:
                pass
            del self.robot_grid_map[robot_id]

    def query_neighbors(self, position, radius):
        """查询指定位置半径内的候选机器人ID"""
        grid_key = tuple((position // self.cell_size).astype(int))
        candidates = []
        
        # 确定搜索范围
        search_range = int(np.ceil(radius / self.cell_size))
        for dx in range(-search_range, search_range+1):
            for dy in range(-search_range, search_range+1):
                neighbor_key = (grid_key[0]+dx, grid_key[1]+dy)
                if neighbor_key in self.grid:
                    candidates.extend(self.grid[neighbor_key])
        return candidates