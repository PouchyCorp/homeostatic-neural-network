from re import match
import pygame
import sys
import numpy as np
import random
import math

TAU = 2 * math.pi

class Homeostat:
    # ---------------- constants ----------------

    UPDATE_RATE = 5

    SENSOR_INTERVAL = 1.0 / 3.0
    DT = 1.0 / UPDATE_RATE

    NUM_UNITS = 2
    NUM_PARAMETERS = 3
    
    STABILITY_THRESHOLD = 3

    # ---------------- init ----------------

    def __init__(self):
        self.num_units = self.NUM_UNITS
        self.num_inputs = self.NUM_UNITS + self.NUM_PARAMETERS

        self.gain = 1.0
        self.upper_bound = 1.0
        self.lower_bound = -1.0

        self.weights = [0.0] * (self.num_inputs * self.num_units)
        self.weight_mutable = [True] * (self.num_inputs * self.num_units)

        self.relay_enabled = [True] * self.num_units

        self.inputs = [0.0] * self.num_inputs
        self.needle_position = [0.0] * self.num_units
        self.needle_velocity = [0.0] * self.num_units
        self.damping = [1.0] * self.num_units

        self.left_eye_angle = math.radians(45)
        self.right_eye_angle = -self.left_eye_angle

        self.crossed_inputs = False

        self.sensor_counter_max = int(self.UPDATE_RATE / self.SENSOR_INTERVAL)
        self.sensor_counter = self.sensor_counter_max

        self.stable_trials = 0
        self.unstable_trials = 0

        self._configure_units()
        self._randomize_weights(self.lower_bound, self.upper_bound)
        self._sever_cross_connections()

    # ---------------- configuration ----------------

    def _configure_units(self):
        for i in range(self.num_units):
            self.set_weight(i, i, -0.5)                    # recurrent
            self.set_weight(self.num_units + 2, i, 1.0)    # essential variable

    def _sever_cross_connections(self):
        self.set_weight(0, 1, 0.0)
        self.set_weight(1, 0, 0.0)

    # ---------------- weights ----------------

    def set_weight(self, input_idx, unit_idx, value):
        idx = input_idx * self.num_units + unit_idx
        self.weights[idx] = value
        self.weight_mutable[idx] = False

    def _randomize_weights(self, low, high):
        for i in range(self.num_inputs):
            for j in range(self.num_units):
                idx = i * self.num_units + j
                if self.weight_mutable[idx]:
                    self.weights[idx] = random.uniform(low, high)

    # ---------------- dynamics ----------------

    def _saturate(self, value):
        if value > self.upper_bound:
            return self.upper_bound
        if value < self.lower_bound:
            return self.lower_bound
        return value

    def integrate_units(self, unit_inputs):
        for i in range(self.num_units):
            dy = self.needle_velocity[i]
            dz = self.gain * unit_inputs[i] - self.damping[i] * self.needle_velocity[i]

            self.needle_position[i] += dy * self.DT
            self.needle_velocity[i] += dz * self.DT

            self.needle_position[i] = self._saturate(self.needle_position[i])
    
    def _reconfig_unit(self, unit_idx):
        for input_idx in range(self.num_inputs):
            w_idx = input_idx * self.num_units + unit_idx
            if self.weight_mutable[w_idx]:
                self.weights[w_idx] = random.uniform(
                    self.lower_bound,
                    self.upper_bound
                )

        # reset unit state
        self.needle_position[unit_idx] = 0.0
        self.needle_velocity[unit_idx] = 0.0
        self.inputs[unit_idx] = 0.0
    
    def reconfig_relays(self):

        for i in range(self.num_units):
            if not self.relay_enabled[i]:
                continue
            
            self._reconfig_unit(i)


    # ---------------- matrix multiply ----------------

    def compute_unit_inputs(self):
        result = [0.0] * self.num_units
        for j in range(self.num_units):
            acc = 0.0
            for i in range(self.num_inputs):
                acc += self.inputs[i] * self.weights[i * self.num_units + j]
            result[j] = acc
        return result

    # ---------------- helpers ----------------

    @staticmethod
    def angle_between(p1, p2):
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0]) % TAU

    @staticmethod
    def distance(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)
    
    @staticmethod
    def read_log():
        try :
            with open("best_performer.txt", "r") as file:
                #read first line to get score
                score_line = file.readline().strip()
                try:
                    score = int(score_line)
                except ValueError:
                    print("warning, log invalid")
                    score = 0
        except FileNotFoundError:
            score = 0
        return score
    
    @staticmethod
    def overwrite_log(score, homeostat):
        print(f"New best performer with score: {score}")
        with open("best_performer.txt", "w") as file:
            file.write(str(score) + "\n")
            file.write(str(homeostat.weights) + "\n") 

    # ---------------- main update ----------------

    def update(self, position, heading, target, world_width):
        theta = (TAU - self.angle_between(position, target)) % TAU

        left_signal = math.cos(theta - heading - self.left_eye_angle)
        right_signal = math.cos(theta - heading - self.right_eye_angle)

        self.distance_signal = self.distance(position, target) / (world_width/2)

        self.inputs[self.num_units] = right_signal if self.crossed_inputs else left_signal
        self.inputs[self.num_units + 1] = left_signal if self.crossed_inputs else right_signal
        self.inputs[self.num_units + 2] = min(self.distance_signal, 1.0)
        unit_inputs = self.compute_unit_inputs()
        self.integrate_units(unit_inputs)

        for i in range(self.num_units):
            self.inputs[i] = self.needle_position[i]
            
        if self.sensor_counter == 0:
            self.stable_trials += 1
            if self.distance_signal > 0.5:
                self.unstable_trials += 1

            if self.unstable_trials > self.STABILITY_THRESHOLD:
                self.reconfig_relays()
                
                # keep track of best performer
                if self.stable_trials > 20:
                    res = self.read_log()
                    if self.stable_trials > res:
                        self.overwrite_log(self.stable_trials, self)

                self.stable_trials = 0
                self.unstable_trials = 0  # instability detected

            self.sensor_counter = self.sensor_counter_max
        else:
            self.sensor_counter -= 1

        return self.inputs
    
    def draw_state(self):
        lines = [
        f"Needle Velocities: {self.needle_velocity}",
        f"Distance: {self.distance_signal}",
        f"Weights: {self.weights[4:]}",
        f"States: {self.inputs[:self.num_units]}",
        f"Right Eye, Left Eye : {self.inputs[self.num_units]}, {self.inputs[self.num_units +1]}",
        f"Stable Trials: {self.stable_trials}, Unstable Trials: {self.unstable_trials}"
        ]

        font = pygame.font.SysFont("Arial", 18)
        y_offset = 10
        for l in lines:
            text_surface = font.render(l, True, (255, 255, 255))
            screen.blit(text_surface, (10, y_offset))
            y_offset += 22




# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1000, 1000
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ashby Mobile Homeostat")

from light import Light
from car import Vehicle

# Main game loop
clock = pygame.time.Clock()
running = True

car = Vehicle()
car.xy = (WIDTH//2 + 100, HEIGHT//2)

light = Light((WIDTH//2, HEIGHT//2), intensity=200.0)

h = Homeostat()

def reset():
    car.xy = (WIDTH//2 + 100, HEIGHT//2)
    car.theta = 0.0
    h.stable_trials = 0
    h.unstable_trials = 0
    h.reconfig_relays()

while running:
    clock.tick(60)

    dt = clock.get_time() / 1000.0
    # Clear screen
    screen.fill((0,0,0))

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
            
        #press R to reset car
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            reset()
        
        if event.type == pygame.KEYDOWN:
            match event.key:
                case pygame.K_KP0:
                    reset()
                    h.weights = [-0.5, 0.0, 0.0, -0.5, -0.096187115, 2.8920174E-4, 0.12929058, 0.07791245, 1.0, 1.0]
                case pygame.K_KP1:
                    reset()
                    h.weights = [-0.5, 0.0, 0.0, -0.5, -0.2475989590137111, 0.7036377667271261, -0.8834543457791277, 0.1790876847115872, 1.0, 1.0]

    h.update(car.xy, car.theta, light.position, world_width=WIDTH)
    
    if h.distance_signal > 2:
        reset()
        h.stable_trials = 0
    
    car.tick(h.inputs[:2], dt)
    car.draw(screen)
    # Draw the light
    light.draw(screen, car)
    #draw faint viability circle
    pygame.draw.circle(screen, (50, 50, 50, 100), (int(light.position[0]), int(light.position[1])), int(WIDTH/4), 1)

    h.draw_state()
    
    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()