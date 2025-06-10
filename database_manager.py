import sqlite3
import os
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_name="traffic_data.db"):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, "data")
        os.makedirs(DATA_DIR, exist_ok=True)

        self.db_path = os.path.join(DATA_DIR, db_name)  # ftll path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_table()

    def _connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
        except sqlite3.Error as e:
            print(f"Connection error: {e}")
            self.conn = None

    def _create_table(self):
        if not self.conn:
            return

        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    density_label TEXT NOT NULL,
                    density_percentage REAL NOT NULL,
                    vehicle_count INTEGER,
                    fps REAL
                );
            ''')
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Creating Errorr: {e}")

    def insert_log(self, density_label, density_percentage, vehicle_count=None, fps=None):
        if not self.conn:
            return
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute('''
                INSERT INTO traffic_logs (timestamp, density_label, density_percentage, vehicle_count, fps)
                VALUES (?, ?, ?, ?, ?);
            ''', (timestamp, density_label, density_percentage, vehicle_count, fps))
            self.conn.commit()

        except sqlite3.Error as e:
            print(f"Insert error: {e} (Debug)")

    def get_logs(self, limit=100):
        if not self.conn:
            print("Cannot retrieve logs.")
            return []
        try:
            self.cursor.execute('SELECT * FROM traffic_logs ORDER BY timestamp DESC LIMIT ?;', (limit,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error fetching logs: {e}")
            return []

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            print("Database connection closed.")