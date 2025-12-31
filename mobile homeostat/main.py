import pygame
import sys
import numpy as np
import random
import math

TAU = 2 * math.pi

class Homeostat:
    # ---------------- constants ----------------

    UPDATE_RATE = 25
    MOTOR_FORCE = 300
    MAX_STABLE_TRIALS = 20

    SENSOR_INTERVAL = 1.0 / 3.0
    DT = 1.0 / UPDATE_RATE

    NUM_UNITS = 2
    NUM_PARAMETERS = 3

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

        self.trials = 0
        self.stable_trials = 0

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

    # ---------------- matrix multiply ----------------

    def compute_unit_inputs(self):
        result = [0.0] * self.num_units
        for j in range(self.num_units):
            acc = 0.0
            for i in range(self.num_inputs):
                acc += self.inputs[i] * self.weights[i * self.num_units + j]
            result[j] = acc
        return result

    # ---------------- relay logic ----------------

    def _reset_unit(self, unit_idx):
        for i in range(self.num_inputs):
            idx = i * self.num_units + unit_idx
            if self.weight_mutable[idx]:
                self.weights[idx] = random.uniform(self.lower_bound, self.upper_bound)

        self.needle_position[unit_idx] = 0.0
        self.needle_velocity[unit_idx] = 0.0
        self.inputs[unit_idx] = 0.0

    def check_relays(self):
        triggered = False
        for i in range(self.num_units):
            if self.relay_enabled[i]:
                y = self.needle_position[i]
                if y <= self.lower_bound or y >= self.upper_bound:
                    self._reset_unit(i)
                    triggered = True
        return triggered

    # ---------------- geometry helpers ----------------

    @staticmethod
    def angle_between(p1, p2):
        return math.atan2(p2[1] - p1[1], p2[0] - p1[0]) % TAU

    @staticmethod
    def distance(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return math.sqrt(dx * dx + dy * dy)

    # ---------------- main update ----------------

    def update(self, position, heading, target, world_width):
        theta = (TAU - self.angle_between(position, target)) % TAU

        left_signal = math.cos(theta - heading - self.left_eye_angle)
        right_signal = math.cos(theta - heading - self.right_eye_angle)

        distance_signal = self.distance(position, target) / (world_width / 2)

        self.inputs[self.num_units] = right_signal if self.crossed_inputs else left_signal
        self.inputs[self.num_units + 1] = left_signal if self.crossed_inputs else right_signal
        self.inputs[self.num_units + 2] = 1.0 if distance_signal > 1.0 else 0.0

        unit_inputs = self.compute_unit_inputs()
        self.integrate_units(unit_inputs)

        for i in range(self.num_units):
            self.inputs[i] = self.needle_position[i]

        # relay check (slow)
        if self.sensor_counter == 0:
            self.trials += 1
            self.stable_trials += 1

            if self.check_relays():
                self.stable_trials = 0

            self.sensor_counter = self.sensor_counter_max
        else:
            self.sensor_counter -= 1

        # motion
        left_speed = self.MOTOR_FORCE * self.inputs[0]
        right_speed = self.MOTOR_FORCE * self.inputs[1]

        angular_change = (right_speed - left_speed) / (2 * world_width)
        forward_speed = (right_speed + left_speed) / 2

        heading = (heading + angular_change * self.DT) % TAU

        dx = forward_speed * math.cos(-heading)
        dy = forward_speed * math.sin(-heading)

        position = (
            position[0] + dx * self.DT,
            position[1] + dy * self.DT,
        )

        return position, heading


# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ashby Mobile Homeostat")

from light import Light
from car import Vehicle

# Main game loop
clock = pygame.time.Clock()
running = True

car = Vehicle()
car.velocity = 2.0  # Set a constant velocity for the car
car.xy = (WIDTH//2 + 10, HEIGHT//2)

light = Light((WIDTH//2 - 100, HEIGHT//2), intensity=200.0)

h = Homeostat()


while running:  
    clock.tick(150)

    # Clear screen
    screen.fill((0,0,0))

    # Handle events -- keep only quit/escape here
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # Update the car's position
    car.xy, car.theta = h.update(car.xy, car.theta, light.position, world_width=100.0)
    
    
    # Draw the car
    car.draw(screen)
    # Draw the light
    light.draw(screen, car)
    
    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()