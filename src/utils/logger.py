import pandas as pd
from datetime import datetime
import os

class SimpleLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.logs = []
    
    def log(self, **kwargs):
        kwargs['timestamp'] = datetime.now()
        self.logs.append(kwargs)
    
    def save(self):
        df = pd.DataFrame(self.logs)
        df.to_csv(self.log_file, index=False)
        print(f"Logs saved to {self.log_file}")