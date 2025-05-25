import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import KDTree
import math
from collections import deque

class Environment:
    def __init__(self, image_name, swarm_size, r_sense, max_steps=200, change_step=200,boundary_mode="potential_field",mean_shift=True):
        self.image_name = image_name
        self.path = rf"image\{image_name}.bmp"
         
        self.frame_time = 0
        self.swarm_size = swarm_size
        self.r_sense = r_sense
        self.max_steps = max_steps
        self.change_step = change_step
        
        self.ratio = 0.1 # 螺旋专用,细线需要调整
        self.sim_scale = 1.5 # 画布放大1.5：single

        if image_name == "grid":
            self.ratio = 0.15 # grid
            self.sim_scale = 1 # 画布放大1.5：single
        elif image_name == "starfish":
            self.ratio = 0.06 # starfish
            self.sim_scale = 1.5 # 画布放大1.5：single
        elif image_name == "Octagonal_star" or image_name == "square":
            if swarm_size <16:
                self.ratio = 0.04
            elif swarm_size in [16,32]:
                self.ratio = 0.06
            elif swarm_size == 64:
                self.ratio = 0.08
            elif swarm_size == 128:
                self.ratio = 0.1
            else:
                self.ratio = 0.12 # starfish
            self.sim_scale = 1.5 # 画布放大1.5：single            
        elif image_name == "snow":
            self.ratio = 0.03 # snow
            self.sim_scale = 1.5 # 画布放大1.5：single
        elif image_name == "spiral_wrap" or image_name == "snow_apart":
            self.ratio = 0.1 # 螺旋专用,细线需要调整
            self.sim_scale = 1 # 画布放大1.5：single
        self.target_area = 1.74 * r_sense * r_sense * swarm_size * self.ratio  # 目标黑色区域面积

        # 比例原来的公式应该是最密六方堆积，3*sqrt(3)/2*r_sense*r_sense*(r_body/r_sense)*ratio 因为实际不会按照通信边缘而按照body,所以进行补偿

        self.img_grid_matrix = self.convert_image_to_grid() # 255: 白色, 0: 黑色

        self.rn = self.img_grid_matrix.shape[0]  # Number of rows
        self.cn = self.img_grid_matrix.shape[1]  # Number of columns
        
        self.labels_num, self.img_grid_matrix_labels = self.cal_image_area_labels()
        self.labels_area = self.cal_labels_area() # 不包含背景

        # 尺寸缩放到合适机器人的规模
        self.scale_factor = np.sqrt(self.target_area/np.sum(self.img_grid_matrix==0))
        self.rn_scaled = self.rn*self.scale_factor
        self.cn_scaled = self.cn*self.scale_factor

        self.labels_area_scaled = self.labels_area*self.scale_factor**2
        self.img_target_area_ratio = np.sum(self.img_grid_matrix==0)/(self.rn*self.cn)
        self.img_target_area_in_scaled = self.img_target_area_ratio*(self.rn_scaled*self.cn_scaled)

        self.img_grid_matrix_labels_to_single = self.multi_labels_matrix_to_single()
        self.full_black_target = self.create_shrunk_matrix(self.img_grid_matrix_labels_to_single.shape, self.img_target_area_ratio) # 一个全黑的目标区域

        # 尺寸缩放到合适机器人的规模(全黑专用)
        self.scale_factor_full_black = np.sqrt(self.target_area/np.sum(self.full_black_target==0))
        self.rn_scaled_full_black = self.rn*self.scale_factor_full_black
        self.cn_scaled_full_black = self.cn*self.scale_factor_full_black

        self.region_gradients = {}  # 新增区域梯度场存储字典
        # key: 区域标签 (1,2,3...)
        # value: dict(grad_x, grad_y, grad_kdtree, grad_data)

        self.boundary_mode = boundary_mode 
        
        # 旋转和偏移
        self.theta = 0/180*np.pi # 逆时针旋转
        self.tx = 0 # 数字增大，原图像左移
        self.ty = 0 # 数字增大，原图像下移

        if self.boundary_mode == "virtual_particles":
            self.dx = 0.45 / self.scale_factor * self.r_sense
            self.virtual_boundary_pos = self.generate_virtual_particles()
        # self.visualize(self.virtual_boundary_pos)
        elif self.boundary_mode == "potential_field":
            self.grad_kdtree, self.grad_data, self.potential =self.generate_potential_field_fast(self.img_grid_matrix, Z=1)
            self.grad_kdtree_full_black, self.grad_data_full_black, self.potential_full_black = self.generate_potential_field_fast(self.full_black_target, Z=1)
            self.generate_labels_potential_field_fast(Z=1)

    def cal_image_area_labels(self):
        binary = np.where(self.img_grid_matrix == 0, 0, 255).astype(np.uint8)
        binary_inv = cv2.bitwise_not(binary) 
        num_labels, labels_im = cv2.connectedComponents(binary_inv, connectivity=8)
        print(f"包含背景，共检测到 {num_labels} 个连通区域")  # 减去1是因为背景也被标记为0
        # 上下左右翻转

        return num_labels, labels_im 

    def cal_labels_area(self):
        labels_area = np.zeros(self.labels_num)
        for label in range(self.labels_num):
                # Count occurrences of the current label in self.img_grid_matrix_labels
            count = np.count_nonzero(self.img_grid_matrix_labels == label)
            labels_area[label] = count
        return labels_area
    
    
    def get_all_contours(self):
        """提取所有轮廓（包括孔洞）"""
        # 转换为OpenCV所需的uint8格式
        mask = (self.img_grid_matrix > 0).astype(np.uint8) * 255
        
        # 使用RETR_TREE检索所有层级轮廓
        contours, hierarchy = cv2.findContours(
            mask, 
            cv2.RETR_TREE, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 分离外层和内层轮廓（根据层级关系）
        outer_contours = []
        inner_contours = []
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:  # 无父轮廓 -> 外层
                outer_contours.append(contours[i])
            else:                          # 有父轮廓 -> 内层（孔洞）
                inner_contours.append(contours[i])
        
        return outer_contours, inner_contours
    
   
    def calculate_labels_area(self):
        """计算每个标签的实际面积"""
        unique, counts = np.unique(self.img_grid_matrix_labels, return_counts=True)
        area_dict = dict(zip(unique, counts))
        return np.array([area_dict.get(i, 0) for i in range(len(area_dict))])
    
    def split_matrix(self, matrix_width, matrix_height, k_list):
        """核心分割算法"""
        regions = []
        remaining_rect = {'x1': 0, 'y1': 0, 'x2': matrix_width, 'y2': matrix_height}
        sum_remaining = 1.0
        
        for k in k_list:
            if sum_remaining <= 0:
                break
            
            p = k / sum_remaining
            w = remaining_rect['x2'] - remaining_rect['x1']
            h = remaining_rect['y2'] - remaining_rect['y1']
            
            # 动态选择分割方向
            if w > h:
                split_value = w * p
                new_x = remaining_rect['x1'] + split_value
                current_region = {
                    'x1': remaining_rect['x1'],
                    'y1': remaining_rect['y1'],
                    'x2': new_x,
                    'y2': remaining_rect['y2']
                }
                remaining_rect['x1'] = new_x
            else:
                split_value = h * p
                new_y = remaining_rect['y1'] + split_value
                current_region = {
                    'x1': remaining_rect['x1'],
                    'y1': remaining_rect['y1'],
                    'x2': remaining_rect['x2'],
                    'y2': new_y
                }
                remaining_rect['y1'] = new_y
            
            regions.append(current_region)
            sum_remaining -= k
        
        return regions
    
    def generate_labels_matrix(self, regions, height, width):
        """根据区域坐标生成标签矩阵"""
        labels_matrix = np.zeros((height, width), dtype=int)
        
        for idx, region in enumerate(regions):
            x1, y1, x2, y2 = region.values()
            
            # 计算列范围（基于中心点）
            col_start = max(0, math.ceil(x1 - 0.5 - 1e-9))
            col_end = min(width, math.floor(x2 - 0.5 + 1e-9) + 1)
            
            # 计算行范围（基于中心点）
            row_start = max(0, math.ceil(y1 - 0.5 - 1e-9))
            row_end = min(height, math.floor(y2 - 0.5 + 1e-9) + 1)
            
            # 索引从1开始
            labels_matrix[row_start:row_end, col_start:col_end] = idx + 1
        
        return labels_matrix
        
    def multi_labels_matrix_to_single(self):
        # 获取图像尺寸
        labels_matrix = self.img_grid_matrix_labels.copy()
        height, width = self.img_grid_matrix_labels.shape

        unique_labels, counts = np.unique(labels_matrix, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))

        # 去掉背景（label 0）
        label_counts.pop(0)

        # 创建一个字典，存储每个区域生长速度的比例
        growth_ratios = {label: count for label, count in label_counts.items()}

        # 使用 deque 作为队列
        queue = deque()

        # 将所有正整数区域的像素位置添加到队列中
        for label, ratio in growth_ratios.items():
            # 获取该区域的位置
            y, x = np.where(labels_matrix == label)
            for i in range(len(x)):
                queue.append((x[i], y[i], label, ratio))

        # 生长过程，向0区域生长
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向
        while queue:
            x, y, label, ratio = queue.popleft()  # 使用 popleft 代替 pop(0)

            for dx, dy in directions:
                nx, ny = x + dx, y + dy

                # 检查新的坐标是否在图像范围内
                if 0 <= nx < width and 0 <= ny < height:
                    if labels_matrix[ny, nx] == 0:  # 如果是0区域，可以生长
                        labels_matrix[ny, nx] = label  # 将0区域替换为当前区域的标签
                        # 按照比例加入队列，以决定生长速度
                        queue.append((nx, ny, label, ratio))

        # 旋转labels_matrix 180度
        
        labels_matrix = labels_matrix[::-1, :]
        


        return labels_matrix



    def sample_contour_points(self, contour):
        """沿单个轮廓采样虚粒子坐标（考虑曲线的长度进行均匀采样）"""
        polygon = contour.squeeze().astype(float)
        points = []

        # 计算轮廓的总长度
        total_length = 0
        lengths = [0]  # 起始点到当前点的累计长度
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]  # 确保闭合
            edge_vec = p2 - p1
            edge_length = np.linalg.norm(edge_vec)
            total_length += edge_length
            lengths.append(total_length)

        # 计算采样点的间隔
        num_samples = int(np.ceil(total_length / self.dx))
        sample_distances = np.linspace(0, total_length, num_samples, endpoint=False)

        # 按照长度采样
        for d in sample_distances:
            # 找到该距离所属的边
            i = np.searchsorted(lengths, d) - 1  # -1 因为 lengths 是从0开始的
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]  # 保证闭合
            
            # 计算该点在线段上的位置
            edge_vec = p2 - p1
            edge_length = np.linalg.norm(edge_vec)
            t = (d - lengths[i]) / edge_length
            
            # 计算该点的位置
            point = p1 + t * edge_vec
            points.append(point)
        
        return np.array(points)
    
    def generate_virtual_particles(self):
        """生成所有虚粒子坐标（物理坐标系）"""
        outer_contours, inner_contours = self.get_all_contours()
        
        all_points = []
        
        # 处理外层轮廓
        for contour in outer_contours:
            points = self.sample_contour_points(contour)
            all_points.extend(points)
        
        # 处理内层孔洞轮廓
        for contour in inner_contours:
            points = self.sample_contour_points(contour)
            all_points.extend(points)
        
        # 转换到物理坐标系
        if self.scale_factor != 1.0:
            all_points = np.array(all_points)
        
        return np.array(all_points)
    
    def convert_image_to_grid(self):
        '''
        图片转栅格并填充为正方形
        '''

        img = Image.open(self.path)  # Open the image file
        img = img.convert('L')  # Convert the image to grayscale ('L' mode)
        img_matrix = np.array(img)  # Convert to NumPy array (matrix)

        # 二值化处理：设置阈值为128，将像素值大于128的设置为白色(255)，小于128的设置为黑色(0)
        # 黑色(0)表示物体，白色(255)表示背景
        _, img_binary = cv2.threshold(img_matrix, 127, 255, cv2.THRESH_BINARY)
        # 填充使图像成为正方形，宽度和高度是原来的self.sim_size倍

        new_dim = int(max(img_binary.shape[1]*self.sim_scale, img_binary.shape[0]*self.sim_scale))
        img_padded = self.pad_to_square(img_binary, new_dim)

        return img_padded

    def create_shrunk_matrix(self, shape, img_target_area_ratio):
        # 计算矩阵的总面积
        total_area = shape[0] * shape[1]
        
        # 计算0和255的数量
        num_zeros = int(total_area * img_target_area_ratio)
        num_255 = total_area - num_zeros
        
        # 初始化全黑矩阵
        matrix = np.zeros(shape, dtype=np.uint8)
        
        # 确定缩小矩阵的中心区域
        center_y, center_x = shape[0] // 2, shape[1] // 2
        
        # 尝试以中心缩小矩阵并填充255的外圈
        # 我们以中心为起点填充0，外围区域逐渐填充255
        radius = 0
        while True:
            # 计算当前半径所覆盖的区域
            top_left = (center_y - radius, center_x - radius)
            bottom_right = (center_y + radius + 1, center_x + radius + 1)
            
            # 当前矩阵中0的数量
            area_of_zeros = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
            
            if area_of_zeros >= num_zeros:
                break
            
            radius += 1
        
        # 填充中心区域为0，外部填充255
        matrix[:top_left[0], :] = 255
        matrix[bottom_right[0]:, :] = 255
        matrix[:, :top_left[1]] = 255
        matrix[:, bottom_right[1]:] = 255
        
        return matrix

    def get_match_pos_in_full_black(self, pos):
        grid_pos = self.robot_pos_to_map_index_scaled_full_black(pos)
        query_pt = np.array([grid_pos[1], grid_pos[0]])  # (x,y)格式

        ref_x = (1-np.sqrt(self.img_target_area_ratio))*self.img_grid_matrix_labels_to_single.shape[0]/2
        ref_y = (1-np.sqrt(self.img_target_area_ratio))*self.img_grid_matrix_labels_to_single.shape[1]/2

        local_x = int((ref_x - query_pt[0])/np.sqrt(self.img_target_area_ratio))
        local_y = int((ref_y - query_pt[1])/np.sqrt(self.img_target_area_ratio))

        # 剪切坐标
        local_x = min(0, max(local_x, -self.img_grid_matrix_labels_to_single.shape[1]+1))
        local_y = min(0, max(local_y, -self.img_grid_matrix_labels_to_single.shape[0]+1))
        print(local_y, -local_x)
        label = self.img_grid_matrix_labels_to_single[local_y, -local_x]

        return label 


    def pad_to_square(self, img_matrix, new_dim):
        '''
        填充图像，使其成为正方形
        '''
        current_height, current_width = img_matrix.shape
        # 计算需要填充的高度和宽度
        pad_top = (new_dim - current_height) // 2
        pad_bottom = new_dim - current_height - pad_top
        pad_left = (new_dim - current_width) // 2
        pad_right = new_dim - current_width - pad_left

        # 使用 np.pad 来填充图像
        padded_img = np.pad(img_matrix, ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=255)  # 填充白色背景 (255)
        return padded_img

    def generate_labels_potential_field_fast(self, Z=1):
        """生成势场及梯度场（包含KDTree构建）"""
        # 原有轮廓处理代码保持不变...
        for label in range(1, self.labels_num):
            # 生成当前区域的二值掩膜
            region_mask = np.where(self.img_grid_matrix_labels == label, 255, 0).astype(np.uint8)

            contours, hierarchy = cv2.findContours(
                region_mask, 
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # 提取区域信息
            height, width = region_mask.shape
            regions = []
            for i, contour in enumerate(contours):
                if hierarchy[0][i][3] != -1:
                    continue
                
                # 计算主轮廓中心
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = (M["m10"]/M["m00"] - width/2)
                cy = (height/2 - M["m01"]/M["m00"])
                
                # 提取主轮廓和孔洞
                all_contours = [contour]
                child_idx = hierarchy[0][i][2]
                while child_idx != -1:
                    all_contours.append(contours[child_idx])
                    child_idx = hierarchy[0][child_idx][0]
                
                # 转换为三维坐标
                region_contours = []
                for cnt in all_contours:
                    points = []
                    for pt in cnt.squeeze():
                        x = pt[0] - width/2
                        y = (height/2 - pt[1])
                        points.append((x, y))
                    region_contours.append(np.array(points))
                
                regions.append({"center": (cx, cy), "contours": region_contours})

            # 生成缩放后的二值图
            canvas = np.zeros_like(region_mask)
            for region in regions:
                for i, contour in enumerate(region["contours"]):
                    # 缩放轮廓
                    scaled = []
                    cx, cy = region["center"]
                    for (x, y) in contour:
                        dx, dy = x - cx, y - cy
                        scaled.append((cx + dx*Z, cy + dy*Z))
                    
                    # 转换回像素坐标
                    pixel_points = []
                    for (x, y) in scaled:
                        px = int(x + width/2)
                        py = int(height/2 - y)
                        pixel_points.append([px, py])
                    
                    # 填充轮廓
                    if i == 0:
                        cv2.fillPoly(canvas, [np.array(pixel_points)], 255)
                    else:
                        cv2.fillPoly(canvas, [np.array(pixel_points)], 0)

            # 计算基础距离场
            dist = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 3)
            
            max_dist = np.max(dist)
            normalized_dist = dist / max_dist
            potential = -dist   # 势场随距离衰减


            # 计算梯度场（改进梯度平滑）
            grad_x = cv2.Sobel(potential, cv2.CV_64F, 1, 0, ksize=11)
            grad_y = -cv2.Sobel(potential, cv2.CV_64F, 0, 1, ksize=11)

            # 梯度归一化与衰减（关键修改点3）
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-5
            attenuation = 1.0 - np.clip(normalized_dist, 0, 1)  # 距离越远衰减越强
            grad_x = (grad_x / grad_magnitude) * attenuation
            grad_y = (grad_y / grad_magnitude) * attenuation
            
            # 梯度场后处理（关键修改点4）
            grad_x = cv2.GaussianBlur(grad_x, (15, 15), sigmaX=3.0)
            grad_y = cv2.GaussianBlur(grad_y, (15, 15), sigmaX=3.0)
            
            """构建梯度查询的KDTree结构"""
            # 生成网格点坐标
            rows, cols = potential.shape
            x_coords = np.arange(cols)
            y_coords = np.arange(rows)
            xx, yy = np.meshgrid(x_coords, y_coords)
            
            # 组合坐标和梯度数据
            points = np.column_stack([xx.ravel(), yy.ravel()])
            gradients = np.column_stack([
                grad_x.ravel(), 
                grad_y.ravel()
            ])

            

            #组合势场数据（新增potential相关存储）
            potential_values = potential.ravel()  # 将势场展平为一维数组

            # 存储区域梯度数据
            self.region_gradients[label] = {
                'grad_x': grad_x,
                'grad_y': grad_y,
                'kdtree': KDTree(points),
                'grad_data': gradients,
                'potential': potential,
                'potential_kdtree': KDTree(points),  # 新增势场KDTree（相同坐标）
                'potential_values': potential_values, # 对应坐标点的势能值

            }

    def get_robot_labels_pos_distance(self, pos, label):
        # 转换为图像像素坐标
        grid_pos = self.robot_pos_to_map_index_scaled(pos)
        query_pt = np.array([grid_pos[1], grid_pos[0]])  # (x,y)格式

        _, idx = self.region_gradients[label]['potential_kdtree'].query(query_pt)
        
        return -self.region_gradients[label]['potential_values'][idx]


    def get_robot_labels_pos_grad(self, pos, label, k=4):
        """
        通过KDTree查询任意点的梯度
        参数:
            pos : tuple/list (x,y) - 查询坐标（图像像素坐标系）
            k : int - 最近邻数量（用于插值时）查询模式 ('linear')
        返回:
            梯度向量 (dx, dy)
        """
        # 转换为图像像素坐标
        grid_pos = self.robot_pos_to_map_index_scaled(pos)
        query_pt = np.array([grid_pos[1], grid_pos[0]])  # (x,y)格式

        grad_kdtree = self.region_gradients[label]['kdtree']
        grad_data = self.region_gradients[label]['grad_data']
    
        # 双线性插值
        dists, indices = grad_kdtree.query(query_pt, k=k)
        
        # 获取周围4个点的坐标和梯度
        neighbor_pts = grad_kdtree.data[indices]
        neighbor_grads = grad_data[indices]
        
        # 计算插值权重（基于距离倒数）
        weights = 1 / (dists + 1e-6)  # 防止除零
        weights /= weights.sum()
        # 加权平均梯度
        grad = np.dot(weights, neighbor_grads)
        
        # 对梯度方向应用旋转修正（关键修复）
        theta = self.theta  # 关键修正：旋转方向取反
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        grad_x = grad[0] * cos_theta + grad[1] * sin_theta  # 修改矩阵乘法顺序
        grad_y = -grad[0] * sin_theta + grad[1] * cos_theta
        
        return np.array((grad_x, grad_y))

    def generate_potential_field_fast(self, matrix, Z):
        """生成势场及梯度场（包含KDTree构建）"""
        # 原有轮廓处理代码保持不变...
        # 检测所有轮廓（包括孔洞）
        _, binary = cv2.threshold(matrix, 127, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(
            binary, 
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 提取区域信息
        height, width = binary.shape
        regions = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] != -1:
                continue
            
            # 计算主轮廓中心
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = (M["m10"]/M["m00"] - width/2)
            cy = (height/2 - M["m01"]/M["m00"])
            
            # 提取主轮廓和孔洞
            all_contours = [contour]
            child_idx = hierarchy[0][i][2]
            while child_idx != -1:
                all_contours.append(contours[child_idx])
                child_idx = hierarchy[0][child_idx][0]
            
            # 转换为三维坐标
            region_contours = []
            for cnt in all_contours:
                points = []
                for pt in cnt.squeeze():
                    x = pt[0] - width/2
                    y = (height/2 - pt[1])
                    points.append((x, y))
                region_contours.append(np.array(points))
            
            regions.append({"center": (cx, cy), "contours": region_contours})

        # 生成缩放后的二值图
        canvas = np.zeros_like(binary)
        for region in regions:
            for i, contour in enumerate(region["contours"]):
                # 缩放轮廓
                scaled = []
                cx, cy = region["center"]
                for (x, y) in contour:
                    dx, dy = x - cx, y - cy
                    scaled.append((cx + dx*Z, cy + dy*Z))
                
                # 转换回像素坐标
                pixel_points = []
                for (x, y) in scaled:
                    px = int(x + width/2)
                    py = int(height/2 - y)
                    pixel_points.append([px, py])
                
                # 填充轮廓
                if i == 0:
                    cv2.fillPoly(canvas, [np.array(pixel_points)], 255)
                else:
                    cv2.fillPoly(canvas, [np.array(pixel_points)], 0)

        # 计算基础距离场
        dist = cv2.distanceTransform(255 - canvas, cv2.DIST_L2, 3)
        
        max_dist = np.max(dist)
        normalized_dist = dist / max_dist
        potential = -dist   # 势场随距离衰减

        grad_x = np.zeros_like(matrix)
        grad_y = np.zeros_like(matrix)

        # 计算梯度场（改进梯度平滑）
        grad_x = cv2.Sobel(potential, cv2.CV_64F, 1, 0, ksize=11)
        grad_y = -cv2.Sobel(potential, cv2.CV_64F, 0, 1, ksize=11)

        # 梯度归一化与衰减（关键修改点3）
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) + 1e-5
        attenuation = 1.0 - np.clip(normalized_dist, 0, 1)  # 距离越远衰减越强
        grad_x = (grad_x / grad_magnitude) * attenuation
        grad_y = (grad_y / grad_magnitude) * attenuation
        
        # 梯度场后处理（关键修改点4）
        grad_x = cv2.GaussianBlur(grad_x, (15, 15), sigmaX=3.0)
        grad_y = cv2.GaussianBlur(grad_y, (15, 15), sigmaX=3.0)
        
        """构建梯度查询的KDTree结构"""
        # 生成网格点坐标
        rows, cols = potential.shape
        x_coords = np.arange(cols)
        y_coords = np.arange(rows)
        xx, yy = np.meshgrid(x_coords, y_coords)
        
        # 组合坐标和梯度数据
        points = np.column_stack([xx.ravel(), yy.ravel()])
        gradients = np.column_stack([
            grad_x.ravel(), 
            grad_y.ravel()
        ])
        
        return KDTree(points), gradients, potential

    def get_robot_pos_grad(self, pos, k=4):
        """
        通过KDTree查询任意点的梯度
        参数:
            pos : tuple/list (x,y) - 查询坐标（图像像素坐标系）
            k : int - 最近邻数量（用于插值时）查询模式 ('linear')
        返回:
            梯度向量 (dx, dy)
        """
        # 转换为图像像素坐标
        grid_pos = self.robot_pos_to_map_index_scaled(pos)
        query_pt = np.array([grid_pos[1], grid_pos[0]])  # (x,y)格式
        
        
        # 双线性插值
        dists, indices = self.grad_kdtree.query(query_pt, k=k)
        
        # 获取周围4个点的坐标和梯度
        neighbor_pts = self.grad_kdtree.data[indices]
        neighbor_grads = self.grad_data[indices]
        
        # 计算插值权重（基于距离倒数）
        weights = 1 / (dists + 1e-6)  # 防止除零
        weights /= weights.sum()
        # 加权平均梯度
        grad = np.dot(weights, neighbor_grads)

        
        # 对梯度方向应用旋转修正（关键修复）
        theta = self.theta  # 关键修正：旋转方向取反
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        grad_x = grad[0] * cos_theta + grad[1] * sin_theta  # 修改矩阵乘法顺序
        grad_y = -grad[0] * sin_theta + grad[1] * cos_theta
        
        return np.array((grad_x, grad_y))
    
    
    def get_robot_pos_grad_full_black(self, pos, k=4):
        """
        通过KDTree查询任意点的梯度
        参数:
            pos : tuple/list (x,y) - 查询坐标（图像像素坐标系）
            k : int - 最近邻数量（用于插值时）查询模式 ('linear')
        返回:
            梯度向量 (dx, dy)
        """
        # 转换为图像像素坐标
        grid_pos = self.robot_pos_to_map_index_scaled_full_black(pos)
        query_pt = np.array([grid_pos[1], grid_pos[0]])  # (x,y)格式
        
        
        # 双线性插值
        dists, indices = self.grad_kdtree_full_black.query(query_pt, k=k)
        
        # 获取周围4个点的坐标和梯度
        neighbor_pts = self.grad_kdtree_full_black.data[indices]
        neighbor_grads = self.grad_data_full_black[indices]
        
        # 计算插值权重（基于距离倒数）
        weights = 1 / (dists + 1e-6)  # 防止除零
        weights /= weights.sum()
        # 加权平均梯度
        grad = np.dot(weights, neighbor_grads)

        
        # 对梯度方向应用旋转修正（关键修复）
        theta = self.theta  # 关键修正：旋转方向取反
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        grad_x = grad[0] * cos_theta + grad[1] * sin_theta  # 修改矩阵乘法顺序
        grad_y = -grad[0] * sin_theta + grad[1] * cos_theta
        
        return np.array((grad_x, grad_y))
    
    def robot_pos_to_map_index_scaled(self, robot_pos):
        x, y = robot_pos
        # 1. 定义画布中心
        cx = self.cn_scaled / 2
        cy = self.rn_scaled / 2
        # 2. 平移到画布中心坐标系
        x_centered = x - cx
        y_centered = y - cy
        # 3. 应用旋转变换（绕中心旋转）
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        x_rot = x_centered * cos_theta - y_centered * sin_theta
        y_rot = x_centered * sin_theta + y_centered * cos_theta
        # 4. 计算旋转后的平移量，并应用
        tx_rot = self.tx * cos_theta - self.ty * sin_theta
        ty_rot = self.tx * sin_theta + self.ty * cos_theta
        # 应用旋转后的平移和中心坐标
        x_transformed = x_rot + cx + tx_rot
        y_transformed = y_rot + cy + ty_rot
        # 5. 处理图像坐标系的Y轴翻转
        flipped_y = self.rn_scaled - y_transformed
        # 6. 边界约束与缩放
        x_grid = min(max(x_transformed, 0), self.cn_scaled - 1) / self.scale_factor
        flipped_y = min(max(flipped_y, 0), self.rn_scaled - 1) / self.scale_factor
        return flipped_y, x_grid
    
    def robot_pos_to_map_index_scaled_full_black(self, robot_pos):
        x, y = robot_pos
        # 1. 定义画布中心
        cx = self.cn_scaled_full_black / 2
        cy = self.rn_scaled_full_black / 2
        # 2. 平移到画布中心坐标系
        x_centered = x - cx
        y_centered = y - cy
        # 3. 应用旋转变换（绕中心旋转）
        cos_theta = np.cos(self.theta)
        sin_theta = np.sin(self.theta)
        x_rot = x_centered * cos_theta - y_centered * sin_theta
        y_rot = x_centered * sin_theta + y_centered * cos_theta
        # 4. 计算旋转后的平移量，并应用
        tx_rot = self.tx * cos_theta - self.ty * sin_theta
        ty_rot = self.tx * sin_theta + self.ty * cos_theta
        # 应用旋转后的平移和中心坐标
        x_transformed = x_rot + cx + tx_rot
        y_transformed = y_rot + cy + ty_rot
        # 5. 处理图像坐标系的Y轴翻转
        flipped_y = self.rn_scaled_full_black - y_transformed
        # 6. 边界约束与缩放
        x_grid = min(max(x_transformed, 0), self.cn_scaled_full_black - 1) / self.scale_factor_full_black
        flipped_y = min(max(flipped_y, 0), self.rn_scaled_full_black - 1) / self.scale_factor_full_black
        return flipped_y, x_grid
    
    def map_index_to_robot_pos_scaled_for_virtual_partical(self):
        """
        栅格索引转机器人坐标（考虑缩放与y轴翻转）
        
        :param grid_row: 栅格行索引（浮点数）
        :param grid_col: 栅格列索引（浮点数）
        :return: (robot_x, robot_y) 机器人坐标
        """
        robot_positions = []
        # 恢复缩放后的坐标
        for grid_row, grid_col in self.virtual_boundary_pos: 
            x_scaled = grid_row * self.scale_factor
            flipped_y_scaled = grid_col * self.scale_factor
            
            # 确保不越界
            x_scaled_clamped = max(0, min(x_scaled, self.cn_scaled - 1))
            flipped_y_scaled_clamped = max(0, min(flipped_y_scaled, self.rn_scaled - 1))
            
            # 计算机器人坐标
            robot_x = x_scaled_clamped
            robot_y = self.rn_scaled - flipped_y_scaled_clamped
            robot_positions.append((robot_x, robot_y))
        
        return robot_positions

    def is_in_target_area(self, robot_pos):
        '''
        return: 0是背景，相同标号表示在同区域
        '''
        grid_pos = self.robot_pos_to_map_index_scaled(robot_pos)
        grid_value = self.img_grid_matrix_labels[int(np.around(grid_pos[0])), int(np.around(grid_pos[1]))]
        
        return grid_value
    
    def set_rotation_and_offset(self, theta, tx, ty):
        self.theta = theta
        self.tx = tx
        self.ty = ty
        


if __name__ == "__main__":

    env = Environment(image_name="round", swarm_size=50, r_sense=5)
    pos = (5, 25)
    print(env.get_robot_pos_grad(pos))
    print(env.get_robot_labels_pos_distance(pos,1))
  
    # plt.figure(figsize=(12, 4))
    
    # plt.subplot(131)

    # for i in range(1, env.labels_num):
    #     plt.imshow(env.region_gradients[i]["potential"], cmap='coolwarm')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(f'output_image_{i}.png', bbox_inches='tight', pad_inches=0)
    # plt.imshow(env.region_gradients[1]["potential"], cmap='viridis')
    plt.imshow(env.potential, cmap='viridis')
    plt.show()
    # grid_pos=env.robot_pos_to_map_index_scaled(pos)
    # # 绘制点
    # plt.scatter(grid_pos[1], grid_pos[0], c='red', s=50, label='Robot Position')
    # # plt.imshow(env.full_black_target, cmap='viridis')
    # plt.title('Potential Field')
    
    # plt.subplot(132)
    # plt.imshow(env.grid_potential, cmap='coolwarm')

    # grid_pos = env.get_robot_pos_in_grid(pos)
    # # 绘制点
    # plt.scatter(grid_pos[1], grid_pos[0], c='red', s=50, label='Robot Position')
    # # plt.imshow(env.region_gradients[1]["grad_x"], cmap='coolwarm')
    # plt.title('X Gradient')
    
    # plt.subplot(133)
    # plt.imshow(env.img_grid_matrix_labels_to_single, cmap='rainbow')
    # plt.title('Y Gradient')
    
    # 去除边框和坐标轴，保存为png
    # 去除坐标轴和边框
    # plt.axis('off')

    # # 保存为 PNG 文件
    # plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)


    # plt.tight_layout()
    # plt.show()

