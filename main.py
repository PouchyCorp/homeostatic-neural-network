import pygame
import sys

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

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
    
    # Fill the screen with white
    screen.fill(WHITE)
    
    # Update the display
    pygame.display.flip()
    
    # Cap the frame rate at 60 FPS
    clock.tick(60)

# Quit pygame
pygame.quit()
sys.exit()