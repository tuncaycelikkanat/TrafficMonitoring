import os
from datetime import datetime

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView, QDateTimeEdit, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QDateTime
from database_manager import DatabaseManager
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


class DataViewerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Traffic Log Viewer")
        self.setGeometry(200, 200, 1000, 700)

        self.export_folder = "export"
        os.makedirs(self.export_folder, exist_ok=True)

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

        self.generate_pdf_button = QPushButton("Create PDF")
        filter_layout.addWidget(self.generate_pdf_button)

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
        self.generate_pdf_button.clicked.connect(self.generate_pdf_report)

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

    def generate_pdf_report(self):
        start_dt_str = self.start_datetime_edit.dateTime().toString("yyyy-MM-dd HH:mm")
        end_dt_str = self.end_datetime_edit.dateTime().toString("yyyy-MM-dd HH:mm")

        table_data = []

        headers = [self.log_table.horizontalHeaderItem(i).text() for i in range(self.log_table.columnCount())]
        table_data.append(headers)

        for row in range(self.log_table.rowCount()):
            row_data = []
            for col in range(self.log_table.columnCount()):
                item = self.log_table.item(row, col)
                if item:
                    row_data.append(item.text())
                else:
                    row_data.append("")
            table_data.append(row_data)

        if len(table_data) <= 1:
            QMessageBox.information(self, "PDF Generation Failed", "No filtered data available to generate a PDF.")
            return

        default_file_name = f"Traffic_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        default_path = os.path.join(self.export_folder, default_file_name)

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save PDF Report",
            default_path,
            "PDF Files (*.pdf);;All Files (*)"
        )

        if not file_name:
            return

        doc = SimpleDocTemplate(file_name, pagesize=letter)
        styles = getSampleStyleSheet()

        story = []

        story.append(Paragraph("<b>Traffic Density Analysis Report</b>", styles['h1']))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph(f"<b>Filter Date Range:</b>", styles['h3']))
        story.append(Paragraph(f"Start: {start_dt_str}", styles['Normal']))
        story.append(Paragraph(f"End: {end_dt_str}", styles['Normal']))
        story.append(Spacer(1, 0.2 * inch))

        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BOX', (0, 0), (-1, -1), 1, colors.black)
        ])

        pdf_table = Table(table_data)
        pdf_table.setStyle(table_style)
        story.append(pdf_table)
        story.append(Spacer(1, 0.5 * inch))

        story.append(
            Paragraph(f"Report Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Italic']))

        try:
            doc.build(story)
            QMessageBox.information(self, "PDF Generated", f"PDF report successfully saved to:\n{file_name}")
            print(f"PDF report saved to: {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "PDF Generation Error", f"An error occurred while generating the PDF report:\n{e}")
            print(f"Error generating PDF: {e}")