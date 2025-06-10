from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QDateTimeEdit, QMessageBox
)
from PyQt5.QtCore import Qt, QDateTime
from database_manager import DatabaseManager


class DataViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Traffic Log Viewer")
        self.setGeometry(200, 200, 1000, 700)

        self.db_manager = DatabaseManager()

        self.setWindowFlags(self.windowFlags())

        self.init_ui()
        self.setup_connections()
        self.load_logs_to_table()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # --- Filters ---
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Start Date/Time:"))
        self.start_datetime_edit = QDateTimeEdit(self)
        self.start_datetime_edit.setCalendarPopup(True)
        self.start_datetime_edit.setDateTime(QDateTime.currentDateTime().addDays(-1))
        self.start_datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        filter_layout.addWidget(self.start_datetime_edit)

        filter_layout.addWidget(QLabel("End Date/Time:"))
        self.end_datetime_edit = QDateTimeEdit(self)
        self.end_datetime_edit.setCalendarPopup(True)
        self.end_datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.end_datetime_edit.setDateTime(QDateTime.currentDateTime())
        filter_layout.addWidget(self.end_datetime_edit)

        self.filter_button = QPushButton("Filter Logs")
        filter_layout.addWidget(self.filter_button)
        filter_layout.addStretch(1)
        main_layout.addLayout(filter_layout)

        self.log_table = QTableWidget()
        main_layout.addWidget(self.log_table)

        self.log_table.setColumnCount(6)
        self.log_table.setHorizontalHeaderLabels([
            "ID", "Timestamp", "Density Label", "Density %", "Vehicle Count", "FPS"
        ])
        self.log_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.log_table.setEditTriggers(QTableWidget.NoEditTriggers)

    def setup_connections(self):
        self.filter_button.clicked.connect(self.load_logs_to_table)

    def load_logs_to_table(self):
        start_dt = self.start_datetime_edit.dateTime()
        end_dt = self.end_datetime_edit.dateTime()

        if start_dt > end_dt:
            QMessageBox.warning(self, "Invalid Date Range",
                                "Start date/time cannot be later than end date/time.")
            return

        start_dt_str = start_dt.toString("yyyy-MM-dd HH:mm") + ":00"
        end_dt_str = end_dt.toString("yyyy-MM-dd HH:mm") + ":59"

        logs = self.db_manager.get_logs_by_time_range(start_dt_str, end_dt_str)

        self.log_table.setRowCount(0)

        if not logs:
            self.log_table.setRowCount(1)
            self.log_table.setSpan(0, 0, 1, self.log_table.columnCount())
            no_data_item = QTableWidgetItem("No data found for the selected date range.")
            no_data_item.setTextAlignment(Qt.AlignCenter)
            self.log_table.setItem(0, 0, no_data_item)
            return

        self.log_table.setRowCount(len(logs))
        for row_idx, log_data in enumerate(logs):
            for col_idx, item_data in enumerate(log_data):
                item = QTableWidgetItem(str(item_data))
                self.log_table.setItem(row_idx, col_idx, item)

        print(f"Loaded {len(logs)} logs from {start_dt_str} to {end_dt_str}.")