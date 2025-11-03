import math
import pygame

RADIAL_GRADIENT = pygame.image.load("radial_gradient.png").convert_alpha()

class Light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity

        self.light_sprite = RADIAL_GRADIENT
        self.light_sprite = pygame.transform.scale_by(
            self.light_sprite, self.intensity * 0.01
        )

    def set_intensity(self, intensity):
        self.intensity = intensity
        self.light_sprite = pygame.transform.scale_by(
            RADIAL_GRADIENT, self.intensity * 0.01
        )

    def draw(self, screen, car):
        
        
        # carve out the area behind the car to simulate shadow
        light_pos = (int(self.position[0] - self.light_sprite.get_width() // 2),
                     int(self.position[1] - self.light_sprite.get_height() // 2))
        light_surface_copy = self.light_sprite.copy()

        #get rotated car polygon
        car_polygon = car.get_polygon()

        # offset car polygon points relative to light position
        car_polygon = [(point[0] - light_pos[0], point[1] - light_pos[1]) for point in car_polygon]

        # raycast from light position to each point in car polygon to create shadow polygon
        shadow_polygon = []
        for point in car_polygon:

            
            direction = pygame.math.Vector2(point[0] - self.position[0], point[1] - self.position[1]).normalize()
            far_point = (point[0] + direction.x * 2000, point[1] + direction.y * 2000)

            shadow_polygon.append(far_point)
            shadow_polygon.append(point)

        

        # sort shadow polygon points clockwise around their centroid
        centroid = pygame.math.Vector2(0,0)
        for point in shadow_polygon:
            centroid += pygame.math.Vector2(point)
        centroid /= len(shadow_polygon) 
        shadow_polygon.sort(key=lambda p: math.atan2(p[1]-centroid.y, p[0]-centroid.x))
        
        pygame.draw.polygon(light_surface_copy, (0,0,0,0), shadow_polygon)
        pygame.draw.polygon(light_surface_copy, (0,0,0,0), car_polygon)
        screen.blit(light_surface_copy, light_pos)

        # draw debug light center
        pygame.draw.circle(screen, (0,255,0), (int(self.position[0]), int(self.position[1])), 10)
    