
from __future__ import annotations
import sys, math, os, threading, concurrent.futures
from typing import Dict, List, Tuple, Callable, Any


import cv2, numpy as np, serial, serial.tools.list_ports
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox, QCheckBox,
    QLineEdit, QFormLayout, QMessageBox, QHBoxLayout, QVBoxLayout, QGroupBox
)

from model import square_model  # 导入模型函数

# ---------- 启用 OpenCV 内部多线程 ----------
cv2.setNumThreads(os.cpu_count() or 4)

# ---------- Worker Thread ----------
class DetectWorker(QThread):
    frameReady = Signal(np.ndarray, list, list)  # frame, corners, ids

    def __init__(self, cam_id: int = 0):
        super().__init__()
        self.cam_id = cam_id
        self.running = False
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_param = cv2.aruco.DetectorParameters()
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_param)
            self.new_api = True
        else:
            self.new_api = False
        cpu_cnt = os.cpu_count() or 4
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(2, cpu_cnt//2))
        self.lock = threading.Lock()
        self.latest_frame = None

    def run(self):
        cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        while self.running:
            ok, frame = cap.read()
            if not ok:
                self.msleep(10)
                continue
            with self.lock:
                self.latest_frame = frame.copy()
            if self.executor._work_queue.qsize() <= 1:
                self.executor.submit(self._detect_latest)
            self.msleep(1)
        cap.release()
        self.executor.shutdown(wait=False)

    def stop(self):
        self.running = False
        self.wait()

    def _detect_latest(self):
        with self.lock:
            frame = self.latest_frame
        if frame is None:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_s = cv2.resize(gray, (320, 240))
        if self.new_api:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray_s)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray_s, self.aruco_dict, parameters=self.aruco_param)
        if corners:
            corners = [c * 2.0 for c in corners]
        self.frameReady.emit(frame, corners if corners else [], [int(i[0]) for i in ids] if ids is not None else [])

# ---------- Main GUI ----------
class RobotGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("多机器人控制 – 修改版")
        self.resize(1200, 640)

        # 在模型参数区
        self.model_func = None                    # 外部模型回调
        self.model_waypoints: Dict[int, List[Tuple[float,float]]] = {}
        self._wp_idx: Dict[int,int] = {}          # 每辆车的当前 waypoint 索引
        self.model_targets: Dict[int, Tuple[float,float]] = {}
        # 模型目标显示开关，默认打开
        self.show_model_targets = True

        # 参数
        self.valid_ids = set(range(16))
        self.max_jump_px = 80
        self.smooth_alpha = 0.3
        self.last_px_pos: Dict[int, Tuple[float, float]] = {}
        self.w, self.h = 640, 480
        self.px_per_m = 1000
        self.robot_paths: Dict[int, List[Tuple[int, int]]] = {}
        self.robots: Dict[int, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
        self.target_world: Tuple[float, float] | None = None
        self.current_ids: set[int] = set()

        # 随机为每个 ID 指定颜色带透明度
        self.colors: Dict[int, QColor] = {i: QColor.fromHsv((i * 30) % 360, 255, 200, 128) for i in range(16)}

        # 用于 waypoint 模式只更新一次
        self._model_waypoint_inited = False

        self._build_ui()
        self._draw_grid()

        self.worker = DetectWorker()
        self.worker.frameReady.connect(self._on_frame)
        self.worker.start()
        self.timer_send = QTimer(self)
        self.timer_send.timeout.connect(self._send_packet)

    def _build_ui(self):
        layout = QHBoxLayout(self)
        # 左：相机界面
        gp_cam = QGroupBox("相机画面")
        vcam = QVBoxLayout(gp_cam)
        self.lbl_cam = QLabel(); self.lbl_cam.setFixedSize(self.w, self.h)
        self.lbl_cam.setStyleSheet("background:black")
        vcam.addWidget(self.lbl_cam)
        self.chk_box = QCheckBox("显示识别框"); self.chk_box.setChecked(True)
        vcam.addWidget(self.chk_box)
        self.chk_detail_cam = QCheckBox("显示详细数据"); self.chk_detail_cam.setChecked(False)
        vcam.addWidget(self.chk_detail_cam)
        layout.addWidget(gp_cam)
        # 右：控制 + 画布
        gp_ctrl = QGroupBox("控制面板")
        vctrl = QVBoxLayout(gp_ctrl)
        self.chk_detail_canvas = QCheckBox("显示详细数据"); self.chk_detail_canvas.setChecked(False)
        vctrl.addWidget(self.chk_detail_canvas)
        # 新增：清空轨迹按钮
        self.btn_clear = QPushButton("清空轨迹")
        self.btn_clear.clicked.connect(self._clear_paths)
        vctrl.addWidget(self.btn_clear)

        self.chk_group = QCheckBox("群控模式"); self.chk_group.setChecked(False)
        vctrl.addWidget(self.chk_group)

        # 新增：模型控制模式
        self.chk_model = QCheckBox("模型控制模式")
        self.chk_model.setChecked(False)
        vctrl.addWidget(self.chk_model)

        # 在“模型控制模式”下面：
        self.chk_show_model_targets = QCheckBox("显示模型目标点")
        self.chk_show_model_targets.setChecked(True)
        vctrl.addWidget(self.chk_show_model_targets)


        self.lbl_canvas = QLabel(); self.lbl_canvas.setFixedSize(self.w, self.h)
        self.lbl_canvas.mousePressEvent = self._canvas_click
        vctrl.addWidget(self.lbl_canvas)
        form = QFormLayout()
        self.cmb_id = QComboBox(); self.cmb_id.addItems([str(i) for i in range(16)])
        form.addRow("机器人ID", self.cmb_id)
        self.le_x = QLineEdit(); self.le_y = QLineEdit(); self.le_speed = QLineEdit("10")
        form.addRow("目标X(m)", self.le_x); form.addRow("目标Y(m)", self.le_y)
        form.addRow("速度(hex)", self.le_speed)
        self.cmb_port = QComboBox(); self._refresh_ports()
        self.cmb_baud = QComboBox(); self.cmb_baud.addItems(["9600", "19220", "38400", "57600", "115200"]); self.cmb_baud.setCurrentText("115200")
        btn_rf = QPushButton("刷新端口"); btn_rf.clicked.connect(self._refresh_ports)
        form.addRow("串口", self.cmb_port); form.addRow("波特率", self.cmb_baud); form.addRow(btn_rf)
        self.btn_open = QPushButton("打开串口"); self.btn_open.clicked.connect(self._toggle_port)
        form.addRow(self.btn_open)
        self.btn_send = QPushButton("开始发送"); self.btn_send.clicked.connect(self._toggle_send)
        form.addRow(self.btn_send)
        vctrl.addLayout(form)
        self.lbl_status = QLabel("状态: 就绪"); vctrl.addWidget(self.lbl_status)
        layout.addWidget(gp_ctrl)

    def _clear_paths(self):
        # 清空所有轨迹但保留当前机器人状态
        self.robot_paths.clear()
        self._paint_canvas()

    def _draw_grid(self):
        self.pm_base = QPixmap(self.w, self.h)
        self.pm_base.fill(Qt.white)
        painter = QPainter(self.pm_base)
        painter.setPen(QPen(QColor('#dddddd')))
        for x in range(0, self.w + 1, 100): painter.drawLine(x, 0, x, self.h)
        for y in range(0, self.h + 1, 100): painter.drawLine(0, y, self.w, y)
        painter.end()
        self.lbl_canvas.setPixmap(self.pm_base)

    def _refresh_ports(self):
        self.cmb_port.clear()
        self.cmb_port.addItems([p.device for p in serial.tools.list_ports.comports()])

    def _toggle_port(self):
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close(); self.btn_open.setText("打开串口"); self.lbl_status.setText("串口已关闭")
        else:
            try:
                self.ser = serial.Serial(self.cmb_port.currentText(), int(self.cmb_baud.currentText()), timeout=1)
                self.btn_open.setText("关闭串口"); self.lbl_status.setText("串口已打开")
            except Exception as e:
                QMessageBox.critical(self, "串口错误", str(e))


    def _toggle_send(self):
        if self.timer_send.isActive():
            # 停止发送
            self.timer_send.stop()
            self.btn_send.setText("开始发送")
            self.lbl_status.setText("发送已停止")
        else:
            # 开始发送
            if not hasattr(self,'ser') or not self.ser.is_open:
                QMessageBox.warning(self,"串口未打开","请先打开串口")
                return
            # 重置 waypoint 初始化标志
            self._model_waypoint_inited = False

            self.timer_send.start(20)
            self.btn_send.setText("停止发送")
            self.lbl_status.setText("发送中…")


    def _send_packet(self):
        # 1) 串口准备
        if not hasattr(self, 'ser') or not self.ser.is_open:
            self._toggle_send()
            return

        # 2) 如果模型模式：
        if self.chk_model.isChecked():
            # 2.1 如果是 waypoint 且还没初始化，就更新一次
            if not self._model_waypoint_inited:
                self._update_model_targets()
                # 如果确实是 waypoint 输出，则标记已初始化
                if self.model_waypoints:
                    self._model_waypoint_inited = True
            # 2.2 对于直接目标模式，我们还是每次都刷新
            elif not self.model_waypoints:
                self._update_model_targets()
                
        # 3) 确定要发送的机器人列表
        if self.chk_group.isChecked():
            targets = list(self.robots.keys())
        else:
            targets = [int(self.cmb_id.currentText())]

        sent = 0
        for rid in targets:
            # 4) 根据当前模式拿 tx, ty
            if self.chk_model.isChecked():
                # a) waypoint 模式
                if rid in self.model_waypoints:
                    idx    = self._wp_idx.get(rid, 0)
                    wp     = self.model_waypoints[rid]
                    # 超出则停在最后一个
                    if idx >= len(wp):
                        tx, ty = wp[-1]
                    else:
                        tx, ty = wp[idx]
                        # 到达容差内就进入下一个
                        rx, ry = self.robots[rid][1]
                        if math.hypot(tx-rx, ty-ry) < 0.02:
                            self._wp_idx[rid] = idx + 1
                # b) 直接目标模式
                elif rid in self.model_targets:
                    tx, ty = self.model_targets[rid]
                else:
                    # 模式下没给目标，就跳过
                    continue
            else:
                # 鼠标模式
                if self.target_world and rid in self.robots:
                    tx, ty = self.target_world
                else:
                    continue

            # 5) 速度
            try:
                spd = int(self.le_speed.text()) & 0xFF
            except ValueError:
                spd = 0

            # 6) 角度都用上面算出的 tx,ty
            rx, ry = self.robots[rid][1]
            target_a = math.degrees(math.atan2(ty - ry, tx - rx))
            # +270 是因为你协议里把 0° 定义在“上方”，顺时针算
            angle = int((target_a + 360 + 270) % 360)

            # 7) 打包并发送
            pkt = bytes([rid, (angle >> 8) & 0xFF, angle & 0xFF, spd])
            try:
                self.ser.write(pkt)
                sent += 1
            except Exception as e:
                QMessageBox.critical(self, "发送失败", str(e))
                self._toggle_send()
                return

        # 8) 状态更新
        self.lbl_status.setText(f"已发送 {sent} 个目标包")


    def _on_frame(self, frame: np.ndarray, corners: list, ids: list):
        # 更新当前 ID 列表
        self.current_ids = set(ids)
        # 移除消失的机器人数据（但保留历史轨迹）
        for rid in list(self.robots.keys()):
            if rid not in self.current_ids:
                self.robots.pop(rid, None)
                self.last_px_pos.pop(rid, None)
        # 绘制目标
        if not self.chk_model.isChecked() and self.target_world:
            tx, ty = self.target_world
            tx_px = int(tx*self.px_per_m + self.w/2)
            ty_px = int(self.h/2 - ty*self.px_per_m)
            cv2.circle(frame, (tx_px, ty_px), 5, (0,0,255), -1)
            if self.chk_detail_cam.isChecked():
                cv2.putText(frame, f"T(px={tx_px},{ty_px}) m({tx:.2f},{ty:.2f})", (tx_px+5, ty_px-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        # 绘制 ArUco
        if self.chk_box.isChecked() and corners:
            ids_np = np.array(ids, dtype=np.int32).reshape(-1,1)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids_np)
        # 显示帧
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.lbl_cam.setPixmap(QPixmap.fromImage(QImage(rgb.data, w,h,w*3, QImage.Format_RGB888)))
        # 更新机器人位置
        for i, corner in enumerate(corners):
            rid = ids[i]
            if rid not in self.valid_ids: continue
            cx, cy = corner[0].mean(axis=0)
            prev = self.last_px_pos.get(rid)
            if prev and np.hypot(cx-prev[0], cy-prev[1]) > self.max_jump_px:
                continue
            if prev and self.smooth_alpha>0:
                cx = prev[0]*(1-self.smooth_alpha) + cx*self.smooth_alpha
                cy = prev[1]*(1-self.smooth_alpha) + cy*self.smooth_alpha
            self.last_px_pos[rid] = (cx, cy)
            wx = (cx - w/2)/self.px_per_m
            wy = (h/2 - cy)/self.px_per_m
            self.robots[rid] = ((cx, cy), (wx, wy))
            self.robot_paths.setdefault(rid, []).append((int(cx), int(cy)))
            # 详细相机标注
            if self.chk_detail_cam.isChecked():
                txt = f"ID{rid} px({int(cx)},{int(cy)}) m({wx:.2f},{wy:.2f})"
                cv2.putText(frame, txt, (int(cx)+5, int(cy)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        # 刷新画布
        self._paint_canvas()

    def _paint_canvas(self):
        pm = self.pm_base.copy()
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1) 画历史路径
        for rid, path in self.robot_paths.items():
            color = self.colors.get(rid, QColor(0,0,255,128))
            pen = QPen(color, 2)
            painter.setPen(pen)
            for pt1, pt2 in zip(path, path[1:]):
                painter.drawLine(pt1[0], pt1[1], pt2[0], pt2[1])

        # 2) 如果是模型模式，画模型目标点；否则画鼠标目标
        if self.chk_model.isChecked() and self.chk_show_model_targets.isChecked():
            for rid, ((cx, cy), (wx, wy)) in self.robots.items():
                # 找下一个目标 tx,ty
                if rid in self.model_waypoints:
                    idx = self._wp_idx.get(rid, 0)
                    pts = self.model_waypoints[rid]
                    # 如果索引越界，就用最后一个点
                    tx, ty = pts[-1] if idx >= len(pts) else pts[idx]
                elif rid in self.model_targets:
                    tx, ty = self.model_targets[rid]
                    print(f"Model target: {tx:.2f}, {ty:.2f}")
                else:
                    continue

                # 世界坐标->像素
                tx_c = int(tx * self.px_per_m + self.w/2)
                ty_c = int(self.h/2 - ty * self.px_per_m)

                # 蓝色目标点
                painter.setPen(Qt.blue)
                painter.setBrush(Qt.blue)
                painter.drawEllipse(tx_c-5, ty_c-5, 10, 10)

                if self.chk_detail_canvas.isChecked():
                    painter.setPen(Qt.black)
                    painter.drawText(tx_c + 8, ty_c,
                                    f"M: ID{rid} px({tx_c},{ty_c}) m({tx:.2f},{ty:.2f})")
        else:
            # 鼠标模式下的红色点
            if self.target_world:
                tx, ty = self.target_world
                tx_c = int(tx * self.px_per_m + self.w/2)
                ty_c = int(self.h/2 - ty * self.px_per_m)
                painter.setPen(Qt.red)
                painter.setBrush(Qt.red)
                painter.drawEllipse(tx_c-5, ty_c-5, 10, 10)
                if self.chk_detail_canvas.isChecked():
                    painter.setPen(Qt.black)
                    painter.drawText(tx_c+8, ty_c,
                                    f"T px({tx_c},{ty_c}) m({tx:.2f},{ty:.2f})")

        # 3) 画机器人当前位置标记
        painter.setFont(QFont('Arial', 10))
        for rid, ((cx, cy), (wx, wy)) in self.robots.items():
            color = self.colors.get(rid, QColor(0, 0, 255, 128))
            painter.setPen(color)
            painter.setBrush(color)
            painter.drawEllipse(int(cx)-5, int(cy)-5, 10, 10)
            if self.chk_detail_canvas.isChecked():
                painter.setPen(Qt.black)
                painter.drawText(int(cx)+8, int(cy), f"ID{rid}")
                painter.drawText(int(cx)+8, int(cy)+12, f"px({int(cx)},{int(cy)})")
                painter.drawText(int(cx)+8, int(cy)+24, f"m({wx:.2f},{wy:.2f})")

        painter.end()
        self.lbl_canvas.setPixmap(pm)


    def _canvas_click(self, event):
        if self.chk_model.isChecked():
            return  # 模型模式下忽略点击
        x = event.position().x(); y = event.position().y()
        wx = (x - self.w/2)/self.px_per_m
        wy = (self.h/2 - y)/self.px_per_m
        self.target_world = (wx, wy)
        self.le_x.setText(f"{wx:.2f}")
        self.le_y.setText(f"{wy:.2f}")
        self._paint_canvas()

    def closeEvent(self, e):
        self.worker.stop()
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        e.accept()

    def set_model(self, func: Callable[[Dict[int,Tuple[float,float]]], Any]):
        """
        func 接收当前机器人位置字典 {id: (wx,wy)}
        func 返回要么 {'waypoints': {id: [(x1,y1),...], ...}}
                要么 {id: (x,y), ...}
        """
        self.model_func = func

    def _update_model_targets(self):
        if not self.model_func:
            return
        # 当前所有机器人位置（世界坐标）
        pos_dict = {rid: data[1] for rid, data in self.robots.items()}
        out = self.model_func(pos_dict)
        print(f"Model output: {out}")
        # 如果有 waypoints
        if isinstance(out, dict) and 'waypoints' in out:
            self.model_waypoints = out['waypoints']
            # 重置所有索引
            self._wp_idx = {rid: 0 for rid in self.model_waypoints}
            # 同时清空单点目标
            self.model_targets.clear()
        else:
            # 普通模式：直接每个 id -> 目标点
            self.model_targets = out
            # 清空 waypoints
            self.model_waypoints.clear()
            self._wp_idx.clear()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = RobotGUI()
    gui.set_model(square_model)
    gui.show()
    sys.exit(app.exec())