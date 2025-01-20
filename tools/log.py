

import numpy as np


class Log:
    def __init__(self):
        self.log = []
    
    def sum(self):
        return np.sum(self.log)
    
    def avg(self):
        return np.mean(self.log)
    
    def append(self, value):
        return self.log.append(value)
    
    def reset(self):
        self.log = []
