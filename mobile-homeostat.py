# mobile homeostat, the agent is a monolithic unit in opposition to ashby's classic homeostat
import pygame as pg
import random
import math
# global vars (may be changed at runtime)
TARGET_MIN = 50
TARGET_MAX = 150
SPEED = 0.1  # how fast the output responds to input


class MobileHomeostat:
    def __init__(self, failure_threshold=40):
        self.failure_threshold = failure_threshold
        self.failures = 0

        self.output : pg.Vector2 = pg.Vector2(0, 0)

    def update(self):
        self.integrate()
        
        if not self.viable():
            self.failures += 1
        
        if self.failures >= self.failure_threshold:
            self.update_weights()
            self.failures = 0
    
    def integrate(self):
        

    def viable(self):
        # viability condition: output magnitude within target range
        mag = self.output.length()
        return TARGET_MIN <= mag <= TARGET_MAX

    def draw(self, screen):
        for unit in self.units:
            unit.draw(screen)


# --- gui ---
