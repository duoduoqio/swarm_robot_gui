import cv2
import numpy as np
import os

class ArucoPoseEstimator:
    def __init__(self, calib_path: str, marker_length: float = 0.1):
        """
        参数:
        - calib_path: 标定文件路径 (npz 文件)，包含 'mtx', 'dist'
        - marker_length: ArUco 标签的边长，单位：米
        """
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"未找到标定文件: {calib_path}")
        data = np.load(calib_path)
        self.camera_matrix = data["mtx"]
        self.dist_coeffs = data["dist"]
        self.marker_length = marker_length

    def estimate(self, frame: np.ndarray, corners: list):
        """
        对检测到的角点进行姿态估计。
        返回:
            - rvecs: 旋转向量 (Nx1x3)
            - tvecs: 平移向量 (Nx1x3)
        """
        if not corners:
            return None, None
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_length,
            self.camera_matrix,
            self.dist_coeffs
        )
        return rvecs, tvecs

    def draw_axes(self, frame: np.ndarray, rvecs, tvecs):
        """
        在图像上绘制坐标轴（红绿蓝: XYZ）
        """
        if rvecs is None or tvecs is None:
            return
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, self.marker_length * 0.5)
