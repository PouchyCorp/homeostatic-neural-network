import pygame
import sys
import numpy as np
import random
import math

class Neuron:
    def __init__(self, n_inputs, mutation_step=0.10):
        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
        self.bias = random.uniform(-1, 1)
        self.output = 0.0
        
        self.mutation_step = mutation_step
        self.last_mutation = None
        self.prev_error = None

    def activate(self, inputs):
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.output = math.tanh(z)
        return self.output

    def _mutate(self):
        """Store a reversible mutation."""
        idx = random.randrange(len(self.weights) + 1)
        delta = random.uniform(-self.mutation_step, self.mutation_step)

        # bias mutation
        if idx == len(self.weights):
            self.bias += delta
            self.last_mutation = ("b", delta)
        else:
            # weight mutation
            self.weights[idx] += delta
            self.last_mutation = ("w", idx, delta)

    def _revert(self):
        """Undo last mutation."""
        if self.last_mutation is None:
            return

        kind = self.last_mutation[0]
        if kind == "b":
            _, delta = self.last_mutation # type: ignore
            self.bias -= delta
        else:
            _, idx, delta = self.last_mutation # type: ignore
            self.weights[idx] -= delta # type: ignore

        self.last_mutation = None

    def adapt(self, error):
        """Ashby-style reversible homeostasis."""
        # first iteration: no previous error to compare
        if self.prev_error is None:
            self.prev_error = error
            self._mutate()
            return

        # If stability improved → keep mutation
        if abs(error) <= abs(self.prev_error):
            # success: forget mutation
            self.last_mutation = None
        else:
            # stability worsened → undo
            self._revert()

        # propose next mutation
        self._mutate()

        self.prev_error = error


class Layer:
    def __init__(self, n_neurons, n_inputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def forward(self, inputs):
        return [n.activate(inputs) for n in self.neurons]

class SimpleNN:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.hidden = Layer(n_hidden, n_inputs)
        self.output = Layer(n_outputs, n_hidden)

    def forward(self, inputs):
        h = self.hidden.forward(inputs)
        return self.output.forward(h)

    def homeostatic_adjustment(self, error):
        for neuron in self.hidden.neurons + self.output.neurons:
            neuron.adapt(error)

    def get_error(self, left_sensor, right_sensor, target):
        total_light = left_sensor + right_sensor
        return total_light - target



# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Homeostatic Neural Network")

from light import Light
import render
from car import Car

# Main game loop
clock = pygame.time.Clock()
running = True

nn = SimpleNN(n_inputs=2, n_hidden=3, n_outputs=1)

frame_count = 0
score_accum = 0.0

car = Car()
car.velocity = 2.0  # Set a constant velocity for the car
car.xy = (WIDTH//2 + 10, HEIGHT//2)

light = Light((WIDTH//2 - 100, HEIGHT//2), intensity=200.0)

target = 0.5  # Target illumination level
previous_error = 0.0
error_accum = 0.0
error = 0.0

iteration_rate = 50  # Adjust every X frames
while running:  
    # Cap the frame rate at 60 FPS
    clock.tick(150)

    # Clear screen
    screen.fill((0,0,0))
    frame_count += 1
    # Handle events -- keep only quit/escape here
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
    
    render.draw_nn(screen, nn, (500, 50))

    #car.theta += 4 * dt  # Slowly rotate the car
    # Update the car's position
    car.tick()

    # Get sensor data
    
    left_sensor, right_sensor = car.get_sensor_data([light])
    sensor_data = [left_sensor, right_sensor]
    nn_outputs = nn.forward(sensor_data)
    error = nn.get_error(left_sensor, right_sensor, target)
    error_accum += error

    car.theta += nn_outputs[0] * 0.05  # Adjust car's angle based on NN output

    if frame_count % iteration_rate == 0:
        error_avg = error_accum / iteration_rate
        error_accum = 0.0
        nn.homeostatic_adjustment(error_avg)

    
    # Draw the car
    car.draw(screen)
    # Draw the light
    light.draw(screen, car)
    # Draw info
    render.draw_text_info(screen, nn, frame_count, left_sensor + right_sensor,
                           target, nn_outputs[0], left_sensor, right_sensor, error)
    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()