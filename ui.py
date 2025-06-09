import cv2
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFrame, QSizePolicy, QLineEdit, QComboBox, QGroupBox
)
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QSize

class TrafficUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traffic Density Analysis System")
        self.setGeometry(100, 100, 1200, 800)
        self.setWindowIcon(QIcon("icon.png"))
        self.setup_ui()
        
    def setup_ui(self):
        # Clean light theme
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))     # Light background
        palette.setColor(QPalette.WindowText, QColor(50, 50, 50))    # Dark text
        palette.setColor(QPalette.Base, QColor(255, 255, 255))       # White elements
        palette.setColor(QPalette.Button, QColor(70, 130, 180))      # SteelBlue
        palette.setColor(QPalette.ButtonText, QColor(255, 255, 255)) # White
        palette.setColor(QPalette.Highlight, QColor(100, 149, 237))  # CornflowerBlue
        self.setPalette(palette)
        
        # Font settings
        title_font = QFont("Arial", 16, QFont.Bold)
        header_font = QFont("Arial", 12, QFont.Bold)
        button_font = QFont("Arial", 10)
        label_font = QFont("Arial", 9)
        
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)
        
        # Title
        title_label = QLabel("TRAFFIC DENSITY ANALYSIS SYSTEM")
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #4682B4; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # Video control panel
        control_layout = QHBoxLayout()
        control_layout.setSpacing(10)
        
        # Model selection
        model_label = QLabel("Model:")
        model_label.setFont(label_font)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Model1", "Model2", "Model3", "Model4", "Model5"])
        self.model_combo.setFont(label_font)
        self.model_combo.setFixedWidth(120)
        
        # Video path
        path_label = QLabel("Video Path:")
        path_label.setFont(label_font)
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Default video will be used")
        self.path_input.setFont(label_font)
        self.path_input.setMinimumWidth(300)
        
        # Status
        self.status_label = QLabel("System ready")
        self.status_label.setFont(label_font)
        self.status_label.setStyleSheet("color: #2E8B57;")
        
        # Buttons
        self.start_button = QPushButton("START")
        self.start_button.setFont(button_font)
        self.start_button.setFixedSize(100, 35)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #32CD32;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #228B22;
            }
        """)
        
        self.stop_button = QPushButton("STOP")
        self.stop_button.setFont(button_font)
        self.stop_button.setFixedSize(100, 35)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background-color: #FF6347;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #CD5C5C;
            }
        """)
        
        self.save_button = QPushButton("SAVE SNAPSHOT")
        self.save_button.setFont(button_font)
        self.save_button.setFixedSize(150, 35)
        self.save_button.setStyleSheet("""
            QPushButton {
                background-color: #4169E1;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1E90FF;
            }
        """)
        
        # Add widgets to control layout
        control_layout.addWidget(model_label)
        control_layout.addWidget(self.model_combo)
        control_layout.addSpacing(10)
        control_layout.addWidget(path_label)
        control_layout.addWidget(self.path_input)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.save_button)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.status_label)
        control_layout.addStretch()
        
        main_layout.addLayout(control_layout)
        
        # Video info
        self.video_info = QLabel("Video info: Not loaded")
        self.video_info.setFont(label_font)
        self.video_info.setStyleSheet("color: #4682B4;")
        main_layout.addWidget(self.video_info)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: #E0E0E0; border: 1px solid #C0C0C0;")
        self.video_label.setMinimumSize(800, 400)
        main_layout.addWidget(self.video_label, 6)  # 60% of space
        
        # Graph layout
        graph_group = QGroupBox("Traffic Density Graph")
        graph_group.setFont(header_font)
        graph_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #C0C0C0;
                border-radius: 5px;
                margin-top: 5px;
                padding-top: 15px;
            }
        """)
        
        self.graph_layout = QVBoxLayout()
        self.graph_layout.setContentsMargins(5, 5, 5, 5)
        graph_group.setLayout(self.graph_layout)
        main_layout.addWidget(graph_group, 4)  # 40% of space
        
        self.setLayout(main_layout)

    def update_image(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Scale to fit label while maintaining aspect ratio
        scaled = qt_image.scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

        self.video_label.setPixmap(QPixmap.fromImage(scaled))
