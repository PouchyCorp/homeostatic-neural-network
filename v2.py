from typing import Optional
import math
import pygame
from homeostat import Homeostat
import random
import render

class Arm:
    def __init__(self, length, angle=0.0, root_start=(0.0, 0.0), is_hand=False, color=(200, 200, 200)):
        self.length = length
        self.angle = angle  # in radians
        self.root_start = root_start
        self.is_hand = is_hand
        self.color = color
        self.child_arms : list['Arm'] = []
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
        
    def draw(self, surface, hand_surface):
        start_x, start_y = self.get_start_position()
        end_x, end_y = self.get_end_position(start_x, start_y)
        pygame.draw.line(surface, self.color, (start_x, start_y), (end_x, end_y), 7)
        if self.is_hand:
            rotated_hand = pygame.transform.rotate(hand_surface, -math.degrees(self.angle)-90)
            surface.blit(rotated_hand, (end_x - rotated_hand.get_width() // 2, end_y - rotated_hand.get_height() // 2))
        for arm in self.child_arms:
            arm.draw(surface, hand_surface)
            
    def add_child(self, arm : 'Arm'):
        self.child_arms.append(arm)
        arm.parent_arm = self
        

if __name__ == "__main__":
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    running = True
    
    HAND_SURFACE = pygame.transform.scale_by(pygame.image.load("hand.png").convert_alpha(), 0.3)

    base_arm = Arm(length=150, angle=math.radians(45), root_start=(400, 300), color=(255,0,0))
    forearm = Arm(length=100, angle=math.radians(30), color=(0,255,0), is_hand=True)
    base_arm.add_child(forearm)
    
    target_points = [(random.randint(0,500), random.randint(0,500))]
    
    homeostat = Homeostat(n_hidden=4, n_outputs=2, zero_init=True, error_scaling=True)
    
    errors = []
    minmaxerror = (10000000000, 0.0)  # large initial range
    while running:
        
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Add target points on numpad key presses
            keys = pygame.key.get_pressed()
            if keys[pygame.K_KP1]:
                target_points[0] = mouse_pos
                
        
            
        outputs = homeostat.forward()
        base_arm.angle = outputs[0] * math.pi
        forearm.angle = outputs[1] * math.pi * 3
        
        error = homeostat.get_error(forearm.get_end_position(*forearm.get_start_position()), target_points[0])
        errors.append(error)
        if error < minmaxerror[0]:
            minmaxerror = (error, minmaxerror[1])
        if error > minmaxerror[1]:
            minmaxerror = (minmaxerror[0], error)
        
        homeostat.homeostatic_adjustment(error, minmaxerror, tolerance=0.1)
        

        screen.fill((30, 30, 30))
        base_arm.draw(screen, HAND_SURFACE)
        for i, target in enumerate(target_points):
            r = min(255, 100 * (i%2) + 100) 
            g = min(255, 50 * (i%3))
            b = min(255, 150 * (i%4))
            pygame.draw.circle(screen, (r, g, b), (int(target[0]), int(target[1])), 15)
        
        
        lines = [
            f"Outputs: [{outputs[0]:.4f}, {outputs[1]:.4f}]",
            f"Target: [{target_points[0][0]:.4f}, {target_points[0][1]:.4f}]",
            f"Error: {error:.4f}",
        ]
        render.draw_text_info(screen, lines)
        
        render.error_graph(screen, errors[:], (10, 400, 200, 150))
        
        render.draw_nn(screen, homeostat, (550, 50), layer_spacing=120, neuron_spacing=50)
            
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()