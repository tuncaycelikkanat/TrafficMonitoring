from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QLineEdit
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import cv2


class TrafficUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Monitoring System")
        self.setGeometry(100, 100, 1280, 800)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                font-family: Arial;
            }
            QPushButton {
                background-color: #ffffff;
                border: 2px solid #ddd;
                border-radius: 10px;
                padding: 15px 30px;
                font-size: 25px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #e6e6e6;
            }
            QLineEdit {
                border: 2px solid #ccc;
                border-radius: 10px;
                padding: 10px;
                font-size: 25px;
                background-color: white;
            }
        """)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            background-color: #ccc;
            border-radius: 15px;
        """)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Video path...")

        #self.model_button = QPushButton("Report")
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        #self.button2 = QPushButton("Save")
        #self.button3 = QPushButton("Exit")

        for b in (self.stop_button, self.start_button):
            b.setCursor(Qt.PointingHandCursor)

        self.button_panel = QFrame()
        self.button_panel.setStyleSheet("background-color: #f0f0f0;")
        self.button_panel.setFixedHeight(160)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(30)
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        self.button_panel.setLayout(button_layout)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        main_layout.addWidget(self.path_input)
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.button_panel)
        self.setLayout(main_layout)

    def update_image(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        scaled_image = qt_image.scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(QPixmap.fromImage(scaled_image))
