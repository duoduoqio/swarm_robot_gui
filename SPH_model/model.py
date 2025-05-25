import numpy as np
from SPH_model.smooth_kernel import *
from env import Environment
from typing import List
from collections import defaultdict

class Robot:
    def __init__(self, R_id, position, velocity, virtual_boundary = False):
        self.R_id = R_id
        self.position = position
        self.velocity = velocity
        self.position_history = []  # 存储历史位置
        self.velocity_history = []  # 存储历史速度
        self.label = 0
        self.neighbors = [] # 有效邻居，包含同区域和背景区域
        self.neighbors_all = [] # 通信内所有邻居
        self.path = []

        self.target_area_label = -1 # 仅在密度平衡中使用

        self.migration_cooldown = 0  # 新增：迁移冷却周期

        # 相关力
        self.f_viscosity = 0
        self.f_pressure = 0
        self.f_extra = np.array([0.0,0.0])

        self.virtual_boundary = virtual_boundary
        if self.virtual_boundary:
            self.rho = 100
        else:
            self.rho = 0

    def update(self, new_pos, new_vel):
        self.position_history.append(new_pos.copy())
        self.velocity_history.append(np.linalg.norm(new_vel))
        # 保持最近100个历史点
        if len(self.position_history) > 100:
            self.position_history.pop(0)
            self.velocity_history.pop(0)
class BaseSPH: 
    '''
    L. C. A. Pimenta et al., “Swarm Coordination Based on Smoothed Particle Hydrodynamics Technique,” IEEE Transactions on Robotics, vol. 29, no. 2, pp. 383–399, Apr. 2013, doi: 10.1109/TRO.2012.2234294.

    '''
    def __init__(self, robot_list:List[Robot], env:Environment, h, max_velocity, dt, boundary=None):
        self.robot_list = robot_list
        self.env = env

        self.boundary = boundary
        self.h = h
        self.rho0 = 1000
        self.k = 100
        self.xi_1 = 1 # 粘性力系数
        self.xi_2 = 2
        self.lam = 7  # 不可压缩水
        self.zeta = 50
        self.m = 1 # 简化模型
        self.dt = dt
        self.k_pressure = 100 # 压力系数

        self.zeta_linear = 200  # 原线性阻尼系数
        self.zeta_nonlinear = 1  # 非线性阻尼系数

        self.max_velocity = max_velocity
        self.max_acceleration = 0.2*max_velocity

        self.change_step = self.env.change_step
    
    def calculate_rho(self):
        for robot in self.robot_list:
            if robot.virtual_boundary:
                continue
            robot.rho = 0
            for neighbor in robot.neighbors:
                r_ij = robot.position - neighbor.position
                distance = np.linalg.norm(r_ij)
                r = distance/self.h
                w = cubic_spline(r, self.h)
                robot.rho += w
            # print("robot.rho",robot.rho,"robot.label",robot.label)
        
    def calculate_pressure_and_viscosity(self):
        for robot in self.robot_list:
            if robot.rho <= 0:
                continue
            robot.f_pressure = 0
            # B_i = 200 * robot.rho*g*H/gamma # g=0.98 H=1/98 gamma=1
            # B_i = 20 * robot.rho/self.lam
            B_i = 20 * robot.rho
            P_i = B_i*((robot.rho/self.rho0)**self.lam-1)      
            c_i = np.sqrt((P_i+B_i)/robot.rho)   
            for neighbor in robot.neighbors:
                v_ij = robot.velocity - neighbor.velocity 
                r_ij = robot.position - neighbor.position
                distance = np.linalg.norm(r_ij)
                r = distance/self.h

                # B_j = 20 * neighbor.rho/self.lam # 此为原式，但效果较差 
                B_j = 20 * neighbor.rho
                P_j = B_j*((neighbor.rho/self.rho0)**self.lam-1)

                w_grad = cubic_spline_grad_zhao(r, self.h)

                if np.dot(v_ij,r_ij)<0:
                    mu_ij = 2*(2+2)*np.dot(v_ij,r_ij)*w_grad/(distance**2+0.01*self.h**2)
                    rho_avg = 0.5*(robot.rho+neighbor.rho)
                    c_j = np.sqrt((P_j+B_j)/neighbor.rho)
                    c_avg = 0.5*(c_i+c_j)

                    viscosity_ij = (-self.xi_1*c_avg*mu_ij+self.xi_2*mu_ij**2)/rho_avg
                else:
                    viscosity_ij = 0

                f_pressure_ij = -(P_i/(robot.rho**2) + P_j/(neighbor.rho**2)+viscosity_ij)*w_grad*(r_ij/distance)
                if neighbor.virtual_boundary:
                    f_pressure_ij = -viscosity_ij*w_grad*(r_ij/distance)

                robot.f_pressure += f_pressure_ij*self.k_pressure

                
 
    def calculate_extra(self):
        if self.env.labels_num >2:
            for robot in self.robot_list:
                if self.env.frame_time < self.change_step:
                    
                    robot.f_extra = np.array([0.0,0.0])
                    map_grad = self.env.get_robot_pos_grad_full_black(robot.position)
                    
                    # self.robot_migration_label(robot)
                    # map_grad = self.process_robot_migration(robot)
                    for neighbor in robot.neighbors:
                        r_ij = robot.position - neighbor.position
                        distance = np.linalg.norm(r_ij)
                        f_extra_ij = self.k*(r_ij/(distance**3) + map_grad)
                        # print("r_ij/(distance**2)",r_ij/(distance**2),"map_grad",map_grad)
                        # print("f_extra_ij",f_extra_ij)
                        robot.f_extra += f_extra_ij
                    if len(robot.neighbors) == 0:
                        map_grad = self.env.get_robot_pos_grad_full_black(robot.position)
                        robot.f_extra = self.k*map_grad
                else:
                    if robot.target_area_label <=0:
                        robot.target_area_label = self.env.get_match_pos_in_full_black(robot.position)

                    robot.f_extra = np.array([0.0,0.0])
                    map_grad = self.env.get_robot_labels_pos_grad(robot.position,robot.target_area_label)
                    
                    # self.robot_migration_label(robot)
                    # map_grad = self.process_robot_migration(robot)
                    for neighbor in robot.neighbors:
                        r_ij = robot.position - neighbor.position
                        distance = np.linalg.norm(r_ij)
                        f_extra_ij = self.k*(r_ij/(distance**3) + map_grad)
                        # print("r_ij/(distance**2)",r_ij/(distance**2),"map_grad",map_grad)
                        # print("f_extra_ij",f_extra_ij)
                        robot.f_extra += f_extra_ij
                    if len(robot.neighbors) == 0:
                        map_grad = self.env.get_robot_labels_pos_grad(robot.position,robot.target_area_label)
                        robot.f_extra = self.k*map_grad
        else:
            for robot in self.robot_list:
                robot.f_extra = np.array([0.0,0.0])
                map_grad = self.env.get_robot_pos_grad(robot.position)
                
                # self.robot_migration_label(robot)
                # map_grad = self.process_robot_migration(robot)
                for neighbor in robot.neighbors:
                    r_ij = robot.position - neighbor.position
                    distance = np.linalg.norm(r_ij)
                    f_extra_ij = self.k*(r_ij/(distance**3) + map_grad)
                    # print("r_ij/(distance**2)",r_ij/(distance**2),"map_grad",map_grad)
                    # print("f_extra_ij",f_extra_ij)
                    robot.f_extra += f_extra_ij
                if len(robot.neighbors) == 0:
                    map_grad = self.env.get_robot_pos_grad(robot.position)
                    robot.f_extra = self.k*map_grad


    def enforce_velocity_limits(self):
        for robot in self.robot_list:
            if robot.virtual_boundary:
                robot.velocity = np.array([0, 0])
            else:
                # 非线性阻尼项：与速度平方成反比
                vel_norm = np.linalg.norm(robot.velocity)
                nonlinear_damp = self.zeta_nonlinear * vel_norm**2
                
                u = (robot.f_pressure + robot.f_extra 
                    - self.zeta_linear * robot.velocity 
                    - nonlinear_damp * robot.velocity/(vel_norm + 1e-6))
                

                norm_u = np.linalg.norm(u)
                if norm_u > self.max_acceleration:
                    u = (u / norm_u) * self.max_acceleration
                robot.velocity += u*self.dt

                if np.linalg.norm(robot.velocity) > self.max_velocity:
                    robot.velocity = robot.velocity/np.linalg.norm(robot.velocity)*self.max_velocity


    def run(self,robot_list):
        self.robot_list = robot_list
        self.calculate_rho()
        self.calculate_pressure_and_viscosity()
        self.calculate_extra()
        
        self.enforce_velocity_limits()


        return self.robot_list
# zhangshuai 10.1016/j.physa.2022.127723
