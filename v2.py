import math
from typing import Optional
import pygame

pygame.init()

class Arm:
    def __init__(self, length, angle=0.0, root_start=(0.0, 0.0), is_hand=False):
        self.length = length
        self.angle = angle  # in radians
        self.root_start = root_start
        self.is_hand = is_hand
        self.child_arms = []
        self.parent_arm : Optional['Arm'] = None

    def get_end_position(self, start_x, start_y):
        end_x = start_x + self.length * math.cos(self.angle)
        end_y = start_y + self.length * math.sin(self.angle)
        return end_x, end_y
    
    def get_start_position(self):
        if self.parent_arm is None:
            return self.root_start
        else:
            parent = self.parent_arm
            return parent.get_end_position(*parent.get_start_position())
        
    def draw(self, surface):
        start_x, start_y = self.get_start_position()
        end_x, end_y = self.get_end_position(start_x, start_y)
        pygame.draw.line(surface, (200, 200, 200), (start_x, start_y), (end_x, end_y), 6)
        if self.is_hand:
            rotated_hand = pygame.transform.rotate(HAND_SURFACE, -math.degrees(self.angle)-90)
            surface.blit(rotated_hand, (end_x - rotated_hand.get_width() // 2, end_y - rotated_hand.get_height() // 2))
        for arm in self.child_arms:
            arm.draw(surface)
            
    def add_child(self, arm : 'Arm'):
        self.child_arms.append(arm)
        arm.parent_arm = self
    

if __name__ == "__main__":
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    running = True
    
    HAND_SURFACE = pygame.transform.scale_by(pygame.image.load("hand.png").convert_alpha(), 0.3)

    base_arm = Arm(length=150, angle=math.radians(45), root_start=(400, 300))
    forearm = Arm(length=100, angle=math.radians(30))
    finger1 = Arm(length=100, angle=math.radians(30), is_hand=True)
    finger2 = Arm(length=80, angle=math.radians(45), is_hand=True)
    finger3 = Arm(length=60, angle=math.radians(60), is_hand=True)
    forearm.add_child(finger1)
    forearm.add_child(finger2)
    forearm.add_child(finger3)
    base_arm.add_child(forearm)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        base_arm.angle += 0.01
        forearm.angle += 0.015
        finger1.angle += 0.02
        finger2.angle += 0.025
        finger3.angle += 0.03

        screen.fill((30, 30, 30))
        base_arm.draw(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()