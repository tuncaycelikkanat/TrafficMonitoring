import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from collections import deque
import numpy as np

class DensityPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        # Create figure with light background
        self.fig = Figure(figsize=(10, 4), dpi=100, facecolor='#FFFFFF')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Configure axes
        self.ax.set_facecolor('#F8F8F8')
        self.ax.tick_params(axis='x', colors='#333333')
        self.ax.tick_params(axis='y', colors='#333333')
        self.ax.spines['bottom'].set_color('#888888')
        self.ax.spines['top'].set_color('#888888') 
        self.ax.spines['right'].set_color('#888888')
        self.ax.spines['left'].set_color('#888888')
        
        # Set titles and labels
        self.ax.set_title("Traffic Density Over Time", 
                         color='#4682B4', fontsize=12)
        self.ax.set_xlabel("Time (seconds)", color='#555555', fontsize=10)
        self.ax.set_ylabel("Density (%)", color='#555555', fontsize=10)
        
        # Set grid and limits
        self.ax.grid(True, linestyle='--', alpha=0.3, color='#CCCCCC')
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 60)
        
        # Initialize data storage
        self.times = deque(maxlen=300)
        self.densities = deque(maxlen=300)
        
        # Create clean plot
        self.line, = self.ax.plot([], [], color='#4169E1', linewidth=2.0, alpha=0.9)
        
        # Add horizontal reference lines
        self.ax.axhline(y=30, color='#32CD32', linestyle='--', alpha=0.5)
        self.ax.axhline(y=60, color='#FFA500', linestyle='--', alpha=0.5)
        self.ax.axhline(y=90, color='#FF6347', linestyle='--', alpha=0.5)
        
        # Add text annotations
        self.ax.text(1, 15, "Low Density", color='#32CD32', fontsize=9)
        self.ax.text(1, 45, "Medium Density", color='#FFA500', fontsize=9)
        self.ax.text(1, 75, "High Density", color='#FF6347', fontsize=9)

    def update_plot(self, current_time, density_percent):
        # Add new data point
        self.times.append(current_time)
        self.densities.append(density_percent)
        
        # Update plot data
        self.line.set_data(self.times, self.densities)
        
        # Adjust view to show the last 60 seconds
        x_min = max(0, current_time - 60)
        x_max = max(60, current_time + 5)  # Add small margin
        
        self.ax.set_xlim(x_min, x_max)
        
        # Efficient redraw
        self.draw_idle()