import math
import pygame

from light import Light

class Car:
    def __init__(self) -> None:
        self.xy = (0,0)
        self.theta = 0.0  # angle in radians
        self.velocity = 0.0  # speed
        self.size = (40, 20)  # width, height
    @property
    def xy(self):
        return (self._x, self._y)
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    @x.setter
    def x(self, val : float):
        self._x = val
    @y.setter
    def y(self, val : float):
        self._y = val
    @xy.setter
    def xy(self, val : tuple[float, float]):
        self._x, self._y = val

    @property
    def direction_vector(self):
        return pygame.math.Vector2(math.cos(self.theta), math.sin(self.theta)).normalize()

    def draw(self, screen):
        empty_surf = pygame.Surface(self.size, pygame.SRCALPHA)
        pygame.draw.rect(empty_surf, (255,0,0), (0,0,self.size[0], self.size[1]))
        rotated_surf = pygame.transform.rotate(empty_surf, -math.degrees(self.theta))
        rotated_surf_center = (rotated_surf.get_width()//2, rotated_surf.get_height()//2)
        screen.blit(rotated_surf, (self.x - rotated_surf_center[0], self.y - rotated_surf_center[1]))

        # draw left and right sensors with small circles
        sensor_offset = 10
        perpendicular = pygame.math.Vector2(-self.direction_vector.y, self.direction_vector.x)
        left_sensor_pos = (self.x - sensor_offset * perpendicular.x, self.y - sensor_offset * perpendicular.y)
        right_sensor_pos = (self.x + sensor_offset * perpendicular.x, self.y + sensor_offset * perpendicular.y)
        pygame.draw.circle(screen, (0,0,255), (int(left_sensor_pos[0]), int(left_sensor_pos[1])), 5)
        pygame.draw.circle(screen, (0,0,255), (int(right_sensor_pos[0]), int(right_sensor_pos[1])), 5)
        
        # Draw direction line
        line_length = 30
        end_x = self.x + self.direction_vector.x * line_length
        end_y = self.y + self.direction_vector.y * line_length
        pygame.draw.line(screen, (0,255,0), (self.x, self.y), (end_x, end_y), 2)

    def tick(self):
        dx = self.velocity * self.direction_vector.x
        dy = self.velocity * self.direction_vector.y
        self.x += dx
        self.y += dy

    def get_sensor_data(self, lights : list[Light]) -> tuple[float, float]:
        """returns the luminance of the left and right sensors, the car blocks light behind it"""
        left_sensor = 0.0
        right_sensor = 0.0
        for light in lights:
            light_pos = pygame.math.Vector2(light.position)
            to_light = light_pos - pygame.math.Vector2(self.xy)
            if to_light.angle_to(self.direction_vector) < 90:  # Light is on the left
                left_sensor += light.intensity / to_light.length() ** 1.25
            else:  # Light is on the right
                right_sensor += light.intensity / to_light.length() ** 1.25
        return (left_sensor, right_sensor)
    
    def get_polygon(self) -> list[tuple[float, float]]:
        """returns the 4 corner points of the rotated car as a list of (x,y) tuples"""
        w, h = self.size
        corners = [
            pygame.math.Vector2(-w/2, -h/2),
            pygame.math.Vector2(w/2, -h/2),
            pygame.math.Vector2(w/2, h/2),
            pygame.math.Vector2(-w/2, h/2),
        ]
        rotated_corners = []
        for corner in corners:
            rotated_x = corner.x * math.cos(self.theta) - corner.y * math.sin(self.theta)
            rotated_y = corner.x * math.sin(self.theta) + corner.y * math.cos(self.theta)
            rotated_corners.append((rotated_x + self.x, rotated_y + self.y))
        return rotated_corners
    


