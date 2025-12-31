import pygame
import random
import math

# ---------------- CONFIG ----------------

N_UNITS = 4
DT = 0.05
FAILURE_THRESHOLD = 40

# asymmetric, constrained ranges (important)
WEIGHT_RANGE = (-2.5, 1.5)
BIAS_RANGE = (-1.0, 1.0)

STATE_MIN = -10.0
STATE_MAX = 10.0

# essential variable viability range
TARGET_MIN = -1.0
TARGET_MAX = 1.0

# relay parameters
RELAY_THRESHOLD = 0.5
SATURATION = 3.0

# inertia
INERTIA = 0.9

# ---------------- CORE ----------------

class Unit:
    def __init__(self, idx, xy):
        self.idx = idx
        self.xy = xy

        # essential variable (what must stay viable)
        self.e = random.uniform(-0.5, 0.5)

        # internal signal (free to vary)
        self.s = random.uniform(-1.0, 1.0)
        self.ds = 0.0

        self.weights = {
            j: random.uniform(*WEIGHT_RANGE)
            for j in range(N_UNITS) if j != idx
        }
        self.bias = random.uniform(*BIAS_RANGE)

    def relay(self, x):
        # crude relay / saturation nonlinearity
        if x > RELAY_THRESHOLD:
            return SATURATION
        elif x < -RELAY_THRESHOLD:
            return -SATURATION
        return 0.0

    def integrate(self):
        # essential variable responds only to internal signal
        self.e += DT * (0.15 * self.s)
    
        # hard clamp
        self.e = max(STATE_MIN, min(STATE_MAX, self.e))


    def regulate(self, signal_vector):
        raw = sum(
            self.weights[j] * signal_vector[j]
            for j in self.weights
        ) + self.bias

        # asymmetric relay with dead zone
        if raw > 0.6:
            target = 2.5
        elif raw < -0.3:
            target = -1.8
        else:
            target = 0.0

        self.ds = INERTIA * self.ds + (1 - INERTIA) * target


    def viable(self):
        return TARGET_MIN <= self.e <= TARGET_MAX

    def reconfigure_one_parameter(self):
        # Ashby-style: change ONE thing, blindly
        choice = random.choice(["weight", "bias"])

        if choice == "bias":
            self.bias = random.uniform(*BIAS_RANGE)
        else:
            j = random.choice(list(self.weights.keys()))
            self.weights[j] = random.uniform(*WEIGHT_RANGE)

    def disturb(self):
        # external disturbance hits essential variable only
        self.e += random.uniform(-1.0, 1.0)


class Homeostat:
    def __init__(self):
        self.units = [
            Unit(i, (200 + (i % 2) * 300, 150 + (i // 2) * 300))
            for i in range(N_UNITS)
        ]
        self.failure_counter = 0

    def step(self):
        signals = [u.s for u in self.units]

        for u in self.units:
            u.regulate(signals)

        for u in self.units:
            u.integrate()

        if not self.viable():
            self.failure_counter += 1
        else:
            self.failure_counter = 0

        if self.failure_counter >= FAILURE_THRESHOLD:
            self.adapt()

    def viable(self):
        return all(u.viable() for u in self.units)

    def adapt(self):
        # random unit, single blind change
        random.choice(self.units).reconfigure_one_parameter()
        self.failure_counter = 0

# ---------------- GUI ----------------

homeostat = Homeostat()
running = True
"""
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Ashby Homeostat")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 22)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            for u in homeostat.units:
                x, y = u.xy
                if (event.pos[0] - x)**2 + (event.pos[1] - y)**2 <= 40**2:
                    u.disturb()

    homeostat.step()

    screen.fill((0, 0, 0))
    for u in homeostat.units:
        color = (0, 180, 0) if u.viable() else (180, 0, 0)
        pygame.draw.circle(screen, color, u.xy, 40)

        txt = font.render(f"e={u.e:.2f}", True, (255, 255, 255))
        screen.blit(txt, (u.xy[0] - 28, u.xy[1] - 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
"""
i = 0
while running:
    i += 1
    homeostat.step()
    # For demonstration purposes, we will just run the simulation without GUI.
    # You can add print statements or logging here to observe the state if needed.
    if i%1000 == 0:
        print(f"Step {i}:")
        for idx, u in enumerate(homeostat.units):
            print(f"  Unit {idx}: e={u.e:.2f}, s={u.s:.2f}, is viable: {u.viable()}")