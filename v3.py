import math
import random
import pygame
from homeostat import Homeostat
import render
from v2 import Arm


    

if __name__ == "__main__":
    screen = pygame.display.set_mode((800, 600))
    clock = pygame.time.Clock()
    running = True
    f = 0
    
    HAND_SURFACE = pygame.transform.scale_by(pygame.image.load("hand.png").convert_alpha(), 0.3)

    base_arm = Arm(length=150, angle=math.radians(45), root_start=(400, 300), color=(255,0,0))
    forearm = Arm(length=100, angle=math.radians(30), color=(0,255,0))
    phalange1 = Arm(length=80, angle=math.radians(20), color=(0,200,200))
    phalange2 = Arm(length=60, angle=math.radians(15), color=(0,150,150))
    phalange3 = Arm(length=100, angle=math.radians(10), color=(0,100,100))
    finger1 = Arm(length=100, angle=math.radians(30), is_hand=True, color=(0,0,255))
    finger2 = Arm(length=80, angle=math.radians(45), is_hand=True, color=(0,0,255))
    finger3 = Arm(length=60, angle=math.radians(60), is_hand=True, color=(0,0,255))
    forearm.add_child(phalange1)
    forearm.add_child(phalange2)
    forearm.add_child(phalange3)
    phalange1.add_child(finger1)
    phalange2.add_child(finger2)
    phalange3.add_child(finger3)
    base_arm.add_child(forearm)
    
    
    
    
    target_points = [(random.randint(0,500), random.randint(0,500)) for _ in range(3)]
    
    minmaxerror1 = (10000000000, 0.0)  # large initial range
    minmaxerror2 = (10000000000, 0.0)  # large initial range
    minmaxerror3 = (10000000000, 0.0)  # large initial range
    
    homeostat = Homeostat(n_hidden=4, n_outputs=2, zero_init=True)
    finger1_homeostat = Homeostat(n_hidden=4, n_outputs=2, zero_init=True, error_scaling=True)
    finger2_homeostat = Homeostat(n_hidden=4, n_outputs=2, zero_init=True, error_scaling=True)
    finger3_homeostat = Homeostat(n_hidden=4, n_outputs=2, zero_init=True, error_scaling=True)
    
    def clamp_minmax(value, current_minmax):
        current_min, current_max = current_minmax
        new_min = min(current_min, value)
        new_max = max(current_max, value)
        return (new_min, new_max)
    
    errors = []

    while running:
        f+=1
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Add target points on numpad key presses
            keys = pygame.key.get_pressed()
            if keys[pygame.K_KP1]:
                target_points[0] = mouse_pos
            if keys[pygame.K_KP2]:
                target_points[1] = mouse_pos
            if keys[pygame.K_KP3]:
                target_points[2] = mouse_pos
                
        
        # two phase updates : the arm perform homeostatic adjustments, then the fingers
        strengh = 2
        if f % 2 == 0:
            outputs_finger1 = finger1_homeostat.forward()
            outputs_finger2 = finger2_homeostat.forward()
            outputs_finger3 = finger3_homeostat.forward()
            finger1.angle = outputs_finger1[0] * math.pi * strengh
            finger2.angle = outputs_finger2[0] * math.pi * strengh
            finger3.angle = outputs_finger3[0] * math.pi * strengh
            phalange1.angle = outputs_finger1[1] * math.pi * strengh
            phalange2.angle = outputs_finger2[1] * math.pi * strengh
            phalange3.angle = outputs_finger3[1] * math.pi * strengh
        else:
            outputs = homeostat.forward()
            base_arm.angle = outputs[0] * math.pi * strengh
            forearm.angle = outputs[1] * math.pi * strengh
        
        error1 = ( homeostat.get_error(finger1.get_end_position(*finger1.get_start_position()), target_points[0]) )
        error2 = ( homeostat.get_error(finger2.get_end_position(*finger2.get_start_position()), target_points[1]) )
        error3 = ( homeostat.get_error(finger3.get_end_position(*finger3.get_start_position()), target_points[2]) )
        minmaxerror1 = clamp_minmax(error1, minmaxerror1)
        minmaxerror2 = clamp_minmax(error2, minmaxerror2)
        minmaxerror3 = clamp_minmax(error3, minmaxerror3)
        error_global = (error1 + error2 + error3)
        errors.append(error_global)
        
        if f % 2 == 0:
            finger1_homeostat.homeostatic_adjustment(error1, minmaxerror1, tolerance=2.0)
            finger2_homeostat.homeostatic_adjustment(error2, minmaxerror2, tolerance=2.0)
            finger3_homeostat.homeostatic_adjustment(error3, minmaxerror3, tolerance=2.0)
        else:
            homeostat.homeostatic_adjustment(error_global, tolerance=5.0)
            
            #special case: it is better for convergence speed if the arm corrects its errors before the fingers are adjusted
            outputs = homeostat.forward()
            base_arm.angle = outputs[0] * math.pi * strengh
            forearm.angle = outputs[1] * math.pi * strengh
            


        screen.fill((30, 30, 30))
        base_arm.draw(screen, HAND_SURFACE)
        for i, target in enumerate(target_points):
            r = min(255, 100 * (i%2) + 100) 
            g = min(255, 50 * (i%3))
            b = min(255, 150 * (i%4))
            pygame.draw.circle(screen, (r, g, b), (int(target[0]), int(target[1])), 15)
            
        lines = [
            f"Outputs: [{base_arm.angle:.4f}, {forearm.angle:.4f}]",
            f"Target1: [{target_points[0][0]:.4f}, {target_points[0][1]:.4f}]",
            f"Target2: [{target_points[1][0]:.4f}, {target_points[1][1]:.4f}]",
            f"Target3: [{target_points[2][0]:.4f}, {target_points[2][1]:.4f}]",
            f"Error: {error_global:.4f}",
            f"Error1: {error1:.4f}",
            f"Error2: {error2:.4f}",
            f"Error3: {error3:.4f}",
        ]
        render.draw_text_info(screen, lines)
        
        render.error_graph(screen, errors[-300:], (10, 400, 200, 50))
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()