import pygame
import sys
import numpy as np
import random
import math

class Neuron:
    def __init__(self, n_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
        self.bias = random.uniform(-1, 1)
        self.output = 0.0

        self.last_bias_change = 0.0  # For homeostatic regulation
        self.last_weight_changes = [0.0 for _ in range(n_inputs)] # For homeostatic regulation

    def activate(self, inputs):
        s = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = math.tanh(s)
        return self.output
    
    def undo_changes(self):
        self.bias -= self.last_bias_change
        for i in range(len(self.weights)):
            self.weights[i] -= self.last_weight_changes[i]

class Layer:
    def __init__(self, n_neurons, n_inputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def forward(self, inputs):
        return [n.activate(inputs) for n in self.neurons]

class SimpleNN:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.hidden = Layer(n_hidden, n_inputs)
        self.output = Layer(n_outputs, n_hidden)

        self.last_score_mean = 0.0

    def forward(self, inputs):
        h = self.hidden.forward(inputs)
        return self.output.forward(h)
    
    def mutate(self, score, threshold=0.9):
        if score >= threshold:
            return # No mutation if performance is good
        
        if score < self.last_score_mean:
            # Revert last changes
            for n in self.hidden.neurons + self.output.neurons:
                n.undo_changes()
        
        mutation_rate = 0.4 * (1 - score)
        mutation_amount = 0.5 * (1 - score)

        for n in self.hidden.neurons + self.output.neurons:
            for i in range(len(n.weights)):
                if random.random() < mutation_rate:
                    n.last_weight_changes[i] = random.uniform(-mutation_amount, mutation_amount)
                    n.weights[i] += n.last_weight_changes[i]    
            if random.random() < mutation_rate:
                n.last_bias_change = random.uniform(-mutation_amount, mutation_amount)
                n.bias += n.last_bias_change

        self.last_score_mean = score
        

class Pendulum:
    def __init__(self, l=2, g=9.81, dt=0.02):
        """
        l: pendulum length (m)
        g: gravity (m/s^2)
        dt: time step (s)
        """
        self.l = l
        self.g = g
        self.dt = dt
        self.base_acc = 0.0  # Horizontal acceleration of the base
        self.reset()


    def reset(self, theta=np.pi/4, theta_dot=0.0, x_base=0.0, x_base_dot=0.0):
        """Reset pendulum and base state."""
        self.theta = theta
        self.theta_dot = theta_dot
        self.x_base = x_base
        self.x_base_dot = x_base_dot

    def step(self):

        self.base_acc = np.clip(self.base_acc, -10.0, 10.0)

        self.x_base_dot += self.base_acc * self.dt
        self.x_base += self.base_acc * self.dt

        self.x_base = np.clip(self.x_base, -3, 3)

        # Pendulum angular acceleration
        theta_ddot = (self.g / self.l) * np.sin(self.theta) + (self.base_acc / self.l) * np.cos(self.theta)

        # Integrate
        self.theta_dot += theta_ddot * self.dt
        self.theta += self.theta_dot * self.dt

        return np.array([
            self.theta, self.theta_dot,
            self.x_base, self.x_base_dot
        ], dtype=np.float32)

    def get_tip_position(self):
        """Return (x, y) coordinates of pendulum tip."""
        x_tip = self.x_base + self.l * np.sin(self.theta)
        y_tip = -self.l * np.cos(self.theta)
        return x_tip, y_tip
    
    def draw(self, screen):
        """Draw the pendulum on the given pygame screen."""
        origin_x = int(self.x_base * 100) + 400  # Scale and center
        origin_y = 300  # Fixed base height
        tip_x, tip_y = self.get_tip_position()
        tip_x = int(tip_x * 100) + 400
        tip_y = int(tip_y * 100) + 300

        # Draw base
        pygame.draw.circle(screen, (0, 0, 0), (origin_x, origin_y), 5)
        # Draw rod
        pygame.draw.line(screen, (0, 0, 255), (origin_x, origin_y), (tip_x, tip_y), 4)
        # Draw tip
        pygame.draw.circle(screen, (255, 0, 0), (tip_x, tip_y), 8)

    def get_score(self):
        """Return score based on pendulum angle (closer to upright is better)."""
        score = np.cos(self.theta)

        # penalize being too far from center
        score -= 0.1 * abs(self.x_base)

        # penalize high angular velocity
        score -= 0.5 * abs(self.theta_dot)
        return score


# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Homeostatic Neural Network")

# Colors
WHITE = (255, 255, 255)

# Main game loop
clock = pygame.time.Clock()
running = True

pendulum = Pendulum()

nn = SimpleNN(n_inputs=4, n_hidden=4, n_outputs=1)

frame_count = 0
score_accum = 0.0

while running:
    # Cap the frame rate at 60 FPS
    clock.tick(150)
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                pendulum.base_acc = -1.0
            elif event.key == pygame.K_RIGHT:
                pendulum.base_acc = 1.0
        else:
            pendulum.base_acc = 0.0
                

        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    delta_time = clock.get_time() / 1000.0  # Convert milliseconds to seconds

    # Update pendulum
    data = pendulum.step()
    
    
    score = pendulum.get_score()
    score_accum += score
    frame_count += 1

    if frame_count >= 75:
        score_mean = score_accum / frame_count
        print(f"Score: {score_mean:.4f}")
        nn.mutate(score_mean)
        frame_count = 0
        score_accum = 0.0

    # Neural network forward pass
    nn_output = nn.forward(data)
    pendulum.base_acc = nn_output[0]

    

    

    # Fill the screen with white
    screen.fill(WHITE)
    # Draw pendulum
    pendulum.draw(screen)
    
    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()