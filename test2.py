import pygame
import math
import random

# ==========================
# CONFIG
# ==========================
SCREEN_W, SCREEN_H = 800, 600
GRAVITY = 0.3
BASE_SPEED = 5
PEND_LENGTH = 150

# ==========================
# HOMEOSTATIC NEURON
# ==========================
class Neuron:
    def __init__(self, num_inputs, learning_rate=0.01, decay=0.999):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-0.5, 0.5)
        self.learning_rate = learning_rate
        self.decay = decay
        self.output = 0.0

    def activate(self, x):
        # Simple tanh activation
        return math.tanh(x)

    def forward(self, inputs):
        # Weighted sum
        z = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = self.activate(z)
        return self.output

    def homeostatic_update(self, error):
        """
        Real-time adaptation: keeps weights stable
        - decay prevents runaway growth
        - bounded updates prevent explosions
        """
        for i in range(len(self.weights)):
            self.weights[i] *= self.decay  # slow decay to prevent explosion
            delta = -self.learning_rate * error * self.output
            self.weights[i] += max(min(delta, 0.05), -0.05)

        self.bias *= self.decay
        self.bias += -self.learning_rate * error * 0.1

# ==========================
# PENDULUM SIM
# ==========================
class PendulumSystem:
    def __init__(self):
        self.base_x = SCREEN_W / 2
        self.angle = random.uniform(-0.5, 0.5)  # radians
        self.angle_vel = 0
        self.neuron = Neuron(2)

    def update(self):
        # Inputs: angle, angular velocity
        output = self.neuron.forward([self.angle, self.angle_vel])

        # Move base according to output, stay in bounds
        self.base_x += output * BASE_SPEED
        self.base_x = max(100, min(SCREEN_W - 100, self.base_x))

        # Physics (simplified)
        torque = GRAVITY * math.sin(self.angle) - 0.05 * self.angle_vel
        self.angle_vel += torque * 0.02 - output * 0.03
        self.angle += self.angle_vel

        # Homeostatic signal: want angle ≈ 0
        error = self.angle
        self.neuron.homeostatic_update(error)

    def draw(self, screen):
        base_y = SCREEN_H - 100
        end_x = self.base_x + math.sin(self.angle) * PEND_LENGTH
        end_y = base_y - math.cos(self.angle) * PEND_LENGTH

        pygame.draw.line(screen, (255, 255, 255), (self.base_x, base_y), (end_x, end_y), 3)
        pygame.draw.circle(screen, (200, 50, 50), (int(end_x), int(end_y)), 10)
        pygame.draw.rect(screen, (80, 80, 80), (self.base_x - 30, base_y, 60, 10))

# ==========================
# MAIN LOOP
# ==========================
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

system = PendulumSystem()
running = True

while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False

    system.update()

    screen.fill((20, 20, 30))
    system.draw(screen)
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
