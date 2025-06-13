import sys
import time
import numpy as np
import cv2
from PIL import Image, ImageDraw
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSlider,
    QPushButton,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import winsound
import pyaudio
import audioop
import numpy as np
import torch


from face import FaceRecognition, FaceManager


class MicrophoneThread(QThread):
    """麦克风输入音量检测线程"""

    volume_updated = pyqtSignal(float)  # 当前音量信号
    wakeup_detected = pyqtSignal()  # 唤醒信号

    def __init__(self, threshold=0.3, parent=None):
        super().__init__(parent)
        self.threshold = threshold
        self.running = True
        self.paused = False
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def run(self):
        """主线程循环"""
        try:
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                frames_per_buffer=1024,
                start=False,
            )
            self.stream.start_stream()

            while self.running:
                if not self.paused:
                    try:
                        data = self.stream.read(1024, exception_on_overflow=False)
                        rms = audioop.rms(data, 2)  # 计算RMS值
                        volume = min(rms / 32767, 1.0)  # 归一化到0-1范围
                        self.volume_updated.emit(volume)
                        if volume > self.threshold:
                            self.wakeup_detected.emit()
                    except Exception as e:
                        print(f"麦克风错误: {e}")
                time.sleep(0.001)
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            self.audio.terminate()

    def stop(self):
        """停止线程"""
        self.running = False
        self.wait()

    def set_threshold(self, threshold):
        """设置唤醒阈值"""
        self.threshold = threshold

    def pause(self):
        """暂停检测"""
        self.paused = True

    def resume(self):
        """恢复检测"""
        self.paused = False


class BuzzerController:
    """独立的蜂鸣器控制类"""

    def __init__(self, cooldown=3):
        self.cooldown = cooldown  # 冷却时间(秒)
        self.last_trigger_time = 0

    def trigger(self):
        """触发蜂鸣器"""
        current_time = time.time()
        if current_time - self.last_trigger_time >= self.cooldown:
            self._beep()
            self.last_trigger_time = current_time
            return True
        return False

    def _beep(self):
        """实际发出蜂鸣声的实现"""
        try:
            print("蜂鸣器触发!")
            winsound.Beep(1000, 500)  # 频率1000Hz，持续时间500ms
        except:
            print("\a")  # 尝试发出系统默认蜂鸣声
            print("蜂鸣器触发!")


class FaceWidget(QLabel):
    """单个脸显示控件"""

    clicked = pyqtSignal(int)

    def __init__(self, face_id, face_img, mark_status, parent=None):
        super().__init__(parent)
        self.face_id = face_id
        self.mark_status = mark_status
        self.setFixedSize(160, 160)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid black;")
        self.update_face(face_img, mark_status)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.face_id)

    def _prepare_image(self, face_img):
        """预处理图像数据"""
        if isinstance(face_img, torch.Tensor):
            face_img = (face_img.permute(1, 2, 0).numpy() + 1) / 2
        return np.ascontiguousarray((face_img * 255).astype(np.uint8))

    def _create_pixmap(self, face_img, mark_status):
        """创建带边框的QPixmap"""
        height, width, _ = face_img.shape
        q_img = QImage(face_img.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # 绘制边框
        painter = QPainter(pixmap)
        color = QColor(255, 0, 0) if mark_status == 1 else QColor(0, 255, 0)
        painter.setPen(QPen(color, 4))
        painter.drawRect(0, 0, width - 1, height - 1)
        painter.end()
        return pixmap

    def update_face(self, face_img, mark_status):
        """更新显示的人脸图像"""
        self.mark_status = mark_status
        face_img = self._prepare_image(face_img)
        pixmap = self._create_pixmap(face_img, mark_status)

        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
        )

    def update_mark_status(self, mark_status):
        """仅更新标记状态而不重新处理图像"""
        if self.mark_status != mark_status:
            self.mark_status = mark_status
            if self.pixmap():
                # 直接在当前pixmap上重绘边框
                pixmap = self.pixmap().copy()
                painter = QPainter(pixmap)
                pen = QPen(
                    QColor(255, 0, 0) if mark_status == 1 else QColor(0, 255, 0), 4
                )
                painter.setPen(pen)
                painter.drawRect(0, 0, pixmap.width() - 1, pixmap.height() - 1)
                painter.end()
                self.setPixmap(pixmap)


class FaceGallery(QWidget):
    """人脸库显示区域"""

    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.face_widgets = {}

    def add_face(self, face_id, face_img, mark_status):
        """添加新人脸"""
        if face_id not in self.face_widgets:
            face_widget = FaceWidget(face_id, face_img, mark_status, self)
            face_widget.clicked.connect(self.main_window.toggle_face_mark)
            self.face_widgets[face_id] = face_widget
            self.layout.addWidget(face_widget)

    def update_face_mark(self, face_id, mark_status):
        """更新人脸标记状态"""
        if face_id in self.face_widgets:
            self.face_widgets[face_id].update_mark_status(mark_status)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能监控 (声音唤醒)")
        self.setGeometry(100, 100, 1200, 600)

        # 初始化模块
        self.face_recognition = FaceRecognition()
        self.face_manager = FaceManager()
        self.buzzer = BuzzerController()  # 独立的蜂鸣器控制器
        self.microphone_thread = MicrophoneThread(threshold=0.3)  # 默认阈值0.3

        # 初始化UI
        self.init_ui()

        # 启动视频定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)  # 约30FPS

        # 启动麦克风线程
        self.microphone_thread.volume_updated.connect(self.update_volume_display)
        self.microphone_thread.wakeup_detected.connect(self.wakeup_system)
        self.microphone_thread.start()

        # 系统状态
        self.system_active = False  # 初始为休眠状态
        self.last_wakeup_time = 0
        self.wakeup_cooldown = 500  # 唤醒后保持激活的秒数

        # 显示初始休眠状态
        self.show_sleep_screen()

    def init_ui(self):
        """初始化用户界面"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid black;")
        self.video_label.setFixedSize(800, 600)
        self.main_layout.addWidget(self.video_label, 3)

        # 右侧控制面板
        self.right_panel = QVBoxLayout()

        # 音量显示
        self.volume_label = QLabel("当前音量: 0.00")
        self.volume_bar = QLabel()
        self.volume_bar.setFixedHeight(20)
        self.volume_bar.setStyleSheet("background-color: green;")

        # 阈值控制
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(30)  # 默认30%
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.threshold_label = QLabel("唤醒阈值: 0.30")

        # 控制按钮
        self.toggle_button = QPushButton("激活系统")
        self.toggle_button.clicked.connect(self.toggle_system)

        # 添加控件到右侧面板
        self.right_panel.addWidget(QLabel("音量监控"))
        self.right_panel.addWidget(self.volume_label)
        self.right_panel.addWidget(self.volume_bar)
        self.right_panel.addWidget(QLabel("唤醒阈值"))
        self.right_panel.addWidget(self.threshold_slider)
        self.right_panel.addWidget(self.threshold_label)
        self.right_panel.addWidget(self.toggle_button)

        # 人脸库区域（可滚动）
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.face_gallery = FaceGallery(self)
        self.scroll_area.setWidget(self.face_gallery)
        self.scroll_area.setMinimumWidth(200)

        self.right_panel.addWidget(QLabel("检测到的人脸"))
        self.right_panel.addWidget(self.scroll_area)

        self.main_layout.addLayout(self.right_panel, 1)

    def show_sleep_screen(self):
        """显示休眠画面"""
        pixmap = QPixmap(self.video_label.size())
        pixmap.fill(Qt.black)
        painter = QPainter(pixmap)
        painter.setPen(Qt.white)
        painter.drawText(
            pixmap.rect(), Qt.AlignCenter, "系统休眠中...\n等待声音唤醒或点击激活按钮"
        )
        painter.end()
        self.video_label.setPixmap(pixmap)

    def update_volume_display(self, volume):
        """更新音量显示"""
        self.volume_label.setText(f"当前音量: {volume:.2f}")

        # 更新音量条
        width = int(volume * 200)
        color = "red" if volume > self.microphone_thread.threshold else "green"
        self.volume_bar.setStyleSheet(f"background-color: {color};")
        self.volume_bar.setFixedWidth(width)

    def update_threshold(self, value):
        """更新唤醒阈值"""
        threshold = value / 100.0
        self.microphone_thread.set_threshold(threshold)
        self.threshold_label.setText(f"唤醒阈值: {threshold:.2f}")

    def wakeup_system(self):
        """唤醒系统"""
        if not self.system_active:
            self.system_active = True
            self.toggle_button.setText("暂停系统")
            self.microphone_thread.resume()
            if not self.face_recognition.open_camera():
                self.show_sleep_screen()
                self.system_active = False
                self.toggle_button.setText("激活系统")
                return
        self.last_wakeup_time = time.time()

    def toggle_system(self):
        """切换系统状态"""
        if self.system_active:
            # 暂停系统
            self.system_active = False
            self.toggle_button.setText("激活系统")
            print("处于关闭系统")
            self.microphone_thread.resume()
            self.face_recognition.release()
            self.show_sleep_screen()
        else:
            # 激活系统
            self.system_active = True
            self.toggle_button.setText("暂停系统")
            print("处于系统开启")
            self.microphone_thread.pause()

            if not self.face_recognition.open_camera():
                self.show_sleep_screen()
                self.system_active = False
                self.toggle_button.setText("激活系统")
                return
            self.last_wakeup_time = time.time()

    def update_frame(self):
        """更新视频帧"""
        # 检查是否需要自动休眠
        if (
            self.system_active
            and time.time() - self.last_wakeup_time > self.wakeup_cooldown
        ):
            self.system_active = False
            self.microphone_thread.pause()
            self.face_recognition.release()
            self.toggle_button.setText("激活系统")
            self.show_sleep_screen()
            return

        if not self.system_active:
            return

        boxes, frame_pil, img_embeddings, face_images = (
            self.face_recognition.process_frame()
        )

        if frame_pil is None:
            return

        # 显示原始帧或处理后的帧
        if boxes is None:
            self.display_frame(np.array(frame_pil))
            return

        # 处理检测到的人脸
        draw = ImageDraw.Draw(frame_pil)

        buzzer_flag = False

        for i, box in enumerate(boxes):
            # 查找或添加人脸
            face_key, prob, mark = self.face_manager.find_face(img_embeddings[i])

            # 检查是否需要触发蜂鸣器
            if mark == 1:
                buzzer_flag = True

            # 更新人脸库 - 传递原始人脸图像而不是特征向量
            if face_key not in self.face_gallery.face_widgets:
                self.face_gallery.add_face(face_key, face_images[i], mark)
            else:
                self.face_gallery.update_face_mark(face_key, mark)

            # 绘制人脸框
            box_color = (255, 0, 0) if mark == 1 else (0, 255, 0)
            draw.rectangle(box.tolist(), outline=box_color, width=6)

            # 添加标签
            text = f"ID: {face_key},   Confidence: {prob:.2f}"
            text_position = (box[0], box[1] - 30)
            text_bbox = draw.textbbox(text_position, text)
            draw.rectangle(text_bbox, fill=(0, 0, 0))
            draw.text(text_position, text, fill=(255, 255, 255))

        if buzzer_flag:
            self.buzzer.trigger()

        # 显示处理后的帧
        self.display_frame(np.array(frame_pil))

    def display_frame(self, frame):
        """显示视频帧"""
        if len(frame.shape) == 3:
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(
                frame.data, width, height, bytes_per_line, QImage.Format_RGB888
            )
        else:
            height, width = frame.shape
            bytes_per_line = width
            q_img = QImage(
                frame.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )
            q_img = q_img.convertToFormat(QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        self.video_label.setPixmap(
            pixmap.scaled(
                self.video_label.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
        )

    def toggle_face_mark(self, face_id):
        """切换人脸标记状态"""
        current_mark = self.face_manager.get_face_mark(face_id)
        new_mark = 1 if current_mark == 0 else 0
        self.face_manager.mark_face(face_id, new_mark)
        self.face_gallery.update_face_mark(face_id, new_mark)

    def closeEvent(self, event):
        """关闭事件处理"""
        self.microphone_thread.stop()
        self.face_recognition.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
