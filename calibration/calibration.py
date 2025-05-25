import cv2
import numpy as np
import glob
import os

# 棋盘格规格（角点数）
CHECKERBOARD = (8, 5)  # 修改为你实际使用的棋盘格角点数
square_size = 24.0  # 每个方格大小，单位：mm

# 终点世界坐标（以棋盘格左上角为原点）
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size  # 转换为实际尺寸单位

# 储存所有图像的角点
objpoints = []  # 3D点
imgpoints = []  # 2D点

# 加载所有标定图像
images = glob.glob('calib_images/*.jpg')  # 你也可以用 png 或其他格式

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        # 提升角点精度
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # 显示角点（可选）
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Corners', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# 相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("相机内参矩阵:\n", mtx)
print("畸变系数:\n", dist)
print("旋转向量:\n", rvecs[0])
print("平移向量:\n", tvecs[0])

# 保存参数
np.savez('camera_calib.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
