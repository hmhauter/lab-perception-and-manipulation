"""
Low-pass filter for the force sensor in the UR flange
"""

class LowPassFilter:
    def __init__(self, alpha):
        self.alpha = alpha
        self.filtered_value = None

    def update(self, new_value):
        if self.filtered_value is None:
            self.filtered_value = new_value
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
