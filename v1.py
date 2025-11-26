import pygame
import sys
import numpy as np
from homeostat import Homeostat

point = [0.0, 0.0]
target = [100.0, 100.0]

def draw_scene(surface, point, target):
    surface.fill((20,20,20))
    pygame.draw.circle(surface, (200,200,50), (int(point[0]), int(point[1])), 10)
    pygame.draw.circle(surface, (50,200,50), (int(target[0]), int(target[1])), 8)

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Homeostatic Neural Network")

# Main game loop
clock = pygame.time.Clock()
running = True

nn = Homeostat(n_hidden=2, n_outputs=2)

import render

# renderer layout constants
NN_ORIGIN = (450, 50)
LAYER_SPACING = 260
NEURON_SPACING = 70

# UI state: output offsets and swap toggle (inputs are unused)
output_offsets = [0.0, 0.0]
const_step = 0.05
swap_outputs = False
# toggle to enable/disable homeostatic adaptation
adapt_enabled = True

# Button layout (created after we know WIDTH/HEIGHT)
btn_w, btn_h = 140, 28
buttons = {
    'add_out0': pygame.Rect(20, HEIGHT - 110, btn_w, btn_h),
    'add_out1': pygame.Rect(20, HEIGHT - 70, btn_w, btn_h),
    'swap_outputs': pygame.Rect(20, HEIGHT - 40, btn_w, btn_h),
}

frame_count = 0
score_accum = 0.0

previous_error = 0.0
error_accum = 0.0
error = 0.0

iteration_rate = 1  # Adjust every X frames
while running:  

    # Clear screen
    screen.fill((0,0,0))
    frame_count += 1

    # Handle events -- keep only quit/escape here
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

        # if R is pressed, randomize weights and biases
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
            nn = Homeostat(n_hidden=2, n_outputs=2)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            # check UI buttons first
            if buttons['add_out0'].collidepoint(mx, my):
                output_offsets[0] += const_step
            elif buttons['add_out1'].collidepoint(mx, my):
                output_offsets[1] += const_step
            elif buttons['swap_outputs'].collidepoint(mx, my):
                swap_outputs = not swap_outputs
            else:
                # check for neuron clicks (use renderer helper to get positions)
                posinfo = render.get_nn_positions(nn, NN_ORIGIN, layer_spacing=LAYER_SPACING, neuron_spacing=NEURON_SPACING)
                handled = False
                for i, (x, y) in enumerate(posinfo['hidden_pos']):
                    r = posinfo.get('hidden_radius', 16)
                    if (mx - x) ** 2 + (my - y) ** 2 <= r * r:
                        # toggle block on corresponding hidden neuron
                        nn.hidden.neurons[i].blocked = not nn.hidden.neurons[i].blocked
                        handled = True
                        break
                if not handled:
                    for i, (x, y) in enumerate(posinfo['output_pos']):
                        r = posinfo.get('output_radius', 20)
                        if (mx - x) ** 2 + (my - y) ** 2 <= r * r:
                            nn.output.neurons[i].blocked = not nn.output.neurons[i].blocked
                            handled = True
                            break
                if not handled:
                    # otherwise set the target position to mouse
                    target[0] = mx
                    target[1] = my
        

    # inputs are irrelevant for this agent; get base outputs and apply perturbations
    base_inputs = [0.0, 0.0]
    raw_nn_output = nn.forward(base_inputs)

    # apply user offsets to outputs
    offset_nn_outputs = [raw_nn_output[0] + output_offsets[0], raw_nn_output[1] + output_offsets[1]]

    # optionally swap both outputs
    if swap_outputs:
        offset_nn_outputs = [offset_nn_outputs[1], offset_nn_outputs[0]]

    point[0] = WIDTH * offset_nn_outputs[0]
    point[1] = HEIGHT * offset_nn_outputs[1]

    error = nn.get_error(point, target)
    # every iteration_rate frames, adjust weights (if adaptation enabled)
    if frame_count % iteration_rate == 0:
        if adapt_enabled:
            nn.homeostatic_adjustment(error)
    draw_scene(screen, point, target)
    
    # draw NN with larger layout and show blocked state; use same layout for positions
    render.draw_nn(screen, nn, NN_ORIGIN, layer_spacing=LAYER_SPACING, neuron_spacing=NEURON_SPACING)
    render.draw_text_info(screen, frame_count, error, [point[0], point[1]], target, raw_nn_output, [target[0]/WIDTH, target[1]/HEIGHT])

    # draw UI buttons
    render.draw_button(screen, buttons['add_out0'], f"Add +{const_step:.2f} to Out0")
    render.draw_button(screen, buttons['add_out1'], f"Add +{const_step:.2f} to Out1")
    swap_label = "Swap Outputs: ON" if swap_outputs else "Swap Outputs: OFF"
    render.draw_button(screen, buttons['swap_outputs'], swap_label, bg=(80,80,100) if swap_outputs else (60,60,60))

    # draw offset status
    ofs_text = f"Out offsets 0={output_offsets[0]:.2f}  1={output_offsets[1]:.2f}"
    ofs_surf = render.FONT_SMALL.render(ofs_text, True, (240,240,240))
    screen.blit(ofs_surf, (20 + btn_w + 10, HEIGHT - 90))
    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()