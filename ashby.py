# Recreation of Ashby's homeostat with gui

import pygame
import random
import math

class Connection:
    def __init__(self, value=0.0) -> None:
        self.value = value

    def get_color(self):
        intensity = int(min(max(self.value * 255, 0), 255))
        return (intensity, intensity, intensity)

class Unit:
    def __init__(self, id: int, xy) -> None:
        self.id = id
        self.xy = xy
        self.value = 0.0
        self.target = 0.0
        self.adaptation_rate = 0.01
        self.weights : dict[int, Connection] = {}
        self.bias = 0.0
    
    def link_unit(self, unit : 'Unit', connection: Connection):
        self.weights[unit.id] = connection
        
    def update_value(self, inputs):
        total_input = sum(w.value * i for w, i in zip(self.weights.values(), inputs)) + self.bias
        self.value = math.tanh(total_input)

    def get_error(self):
        return abs(self.target - self.value)
    
    def is_stable(self):
        return self.get_error() < 0.1
    
    def adapt(self):
        if not self.is_stable():
            self.adaptation_rate = min(0.1, self.adaptation_rate * 1.05)
            for k in self.weights:
                self.weights[k].value += random.uniform(-self.adaptation_rate, self.adaptation_rate)
            self.bias += random.uniform(-self.adaptation_rate, self.adaptation_rate)
        else:
            self.adaptation_rate = max(0.001, self.adaptation_rate * 0.95)

class Homeostat:
    def __init__(self, n_units: int):
        self.units = [Unit(i, (100 + (i % 2) * 300, 100 + (i // 2) * 300)) for i in range(n_units)]


        self.connections = []

        connected : set[tuple[int, int]] = set()
        for unit in self.units:
            for other_unit in self.units:
                if (unit.id, other_unit.id) in connected or (other_unit.id, unit.id) in connected or unit.id == other_unit.id:
                    continue
                connection = Connection(random.uniform(-1.0, 1.0))
                unit.link_unit(other_unit, connection)
                other_unit.link_unit(unit, connection)
                self.connections.append((unit, other_unit, connection))
                connected.add((unit.id, other_unit.id))

    def step(self):
        inputs = [unit.value for unit in self.units]
        for unit in self.units:
            unit.update_value(inputs)
    
    def adapt(self):
        for unit in self.units:
            unit.adapt()
    
    def get_overall_error(self):
        return sum(unit.get_error() for unit in self.units) / len(self.units)
    
    def draw(self, surface, origin):

        # draw line connections between units depending on weights
        for unit_a, unit_b, connection in self.connections:
            x1, y1 = unit_a.xy
            x2, y2 = unit_b.xy
            color = connection.get_color()
            pygame.draw.line(surface, (128, 128, 128), (x1, y1), (x2, y2), 7)
            pygame.draw.line(surface, color, (x1, y1), (x2, y2), 5)
            two_third = ( (x1 * 2 + x2) // 3, (y1 * 2 + y2) // 3 )
            font = pygame.font.SysFont(None, 24)
            text = font.render(f"{connection.value:.2f}", True, (255, 255, 255))
            surface.blit(text, (two_third[0] - 15, two_third[1] - 10))
            


        for i, unit in enumerate(self.units):
            x, y = unit.xy
            color = (0, 255, 0) if unit.is_stable() else (255, 0, 0)
            pygame.draw.circle(surface, color, (x, y), 50)
            font = pygame.font.SysFont(None, 24)
            text = font.render(f"{unit.value:.2f}", True, (255, 255, 255))
            bias = font.render(f"B:{unit.bias:.2f}", True, (255, 255, 255))
            surface.blit(bias, (x - 15, y + 10))
            surface.blit(text, (x - 15, y - 10))

    def click_unit(self, pos):
        for unit in self.units:
            x, y = unit.xy
            if (pos[0] - x) ** 2 + (pos[1] - y) ** 2 <= 30 ** 2:
                unit.value = random.uniform(0.0, 1.0)


# Pygame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ashby's Homeostat Simulation")
clock = pygame.time.Clock()
homeostat = Homeostat(n_units=4)
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                homeostat.click_unit(event.pos)

    homeostat.step()
    homeostat.adapt()

    screen.fill((0, 0, 0))
    homeostat.draw(screen, (100, 100))

    overall_error = homeostat.get_overall_error()
    font = pygame.font.SysFont(None, 36)
    error_text = font.render(f"Overall Error: {overall_error:.4f}", True, (255, 255, 255))
    screen.blit(error_text, (10, 10))

    pygame.display.flip()
    clock.tick(5)
    
    