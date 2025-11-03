import pygame
import sys
import numpy as np
import random
import math

class Neuron:
    def __init__(self, n_inputs, mutation_step=0.01, improvement_tolerance=1.0):
        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
        self.bias = random.uniform(-1, 1)
        self.output = 0.0
        
        self.mutation_step = mutation_step
        self.last_mutation = None
        # persistent mutation direction: +1 means increase, -1 means decrease
        # while a sequence of successful mutations continues, keep this sign
        self.mutation_direction = random.choice([-1, 1])
        # how much lower the new error must be compared to prev_error to be
        # considered a clear improvement (no mutation while this holds)
        self.improvement_tolerance = improvement_tolerance
        self.prev_error = None

    def activate(self, inputs):
        z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.output = math.tanh(z)
        return self.output

    def _mutate(self):
        """Store a reversible mutation."""
        idx = random.randrange(len(self.weights) + 1)
        # Ensure we mutate in the current persistent direction (sign)
        sign = self.mutation_direction if self.mutation_direction in (1, -1) else random.choice([-1, 1])
        # small random magnitude, but fixed sign
        delta = sign * random.uniform(0, self.mutation_step)

        # bias mutation
        if idx == len(self.weights):
            self.bias += delta
            self.last_mutation = ("b", delta)
        else:
            # weight mutation
            self.weights[idx] += delta
            self.last_mutation = ("w", idx, delta)

    def _revert(self):
        """Undo last mutation."""
        if self.last_mutation is None:
            return

        kind = self.last_mutation[0]
        if kind == "b":
            _, delta = self.last_mutation # type: ignore
            self.bias -= delta
        else:
            _, idx, delta = self.last_mutation # type: ignore
            self.weights[idx] -= delta # type: ignore

        self.last_mutation = None

    def adapt(self, error):
        """Ashby-style reversible homeostasis."""
        # first iteration: no previous error to compare
        if error < 5:
            return  # no adaptation needed for low error

        if self.prev_error is None:
            self.prev_error = error
            self._mutate()
            return

        # delta = new - old; negative means improvement (lower error)
        delta = error - self.prev_error

        # If there's a clear improvement (strictly negative beyond tolerance),
        # commit the last mutation (if any) and do NOT propose a new mutation.
        if delta < -self.improvement_tolerance:
            # commit the mutation
            self.last_mutation = None
            # update prev_error and skip proposing further mutations while
            # error keeps improving
            self.prev_error = error
            return

        # Otherwise, treat according to improvement/worsening as before.
        # If stability improved → keep mutation
        if error <= self.prev_error:
            # success: commit mutation (forget reversible record)
            self.last_mutation = None
        else:
            # stability worsened → undo and flip direction for next exploration
            self._revert()
            self.mutation_direction *= -1

        # propose next mutation
        self._mutate()

        self.prev_error = error


class Layer:
    def __init__(self, n_neurons, n_inputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def forward(self, inputs):
        return [n.activate(inputs) for n in self.neurons]

class SimpleNN:
    def __init__(self, n_inputs, n_hidden, n_outputs):
        self.hidden = Layer(n_hidden, n_inputs)
        self.output = Layer(n_outputs, n_hidden)

    def forward(self, inputs):
        h = self.hidden.forward(inputs)
        return self.output.forward(h)

    def homeostatic_adjustment(self, error):
        for neuron in self.hidden.neurons + self.output.neurons:
            neuron.adapt(error)

    def get_error(self, point, target):
        return math.hypot(point[0]-target[0], point[1]-target[1])

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

nn = SimpleNN(n_inputs=2, n_hidden=3, n_outputs=2)

import render

# UI state: input offsets and swap toggle
input_offsets = [0.0, 0.0]
const_step = 0.05
swap_inputs = False
# toggle to enable/disable homeostatic adaptation
adapt_enabled = True

# Button layout (created after we know WIDTH/HEIGHT)
btn_w, btn_h = 140, 28
buttons = {
    'pause_adapt': pygame.Rect(20, HEIGHT - 150, btn_w, btn_h),
    'add_left': pygame.Rect(20, HEIGHT - 110, btn_w, btn_h),
    'add_right': pygame.Rect(20, HEIGHT - 70, btn_w, btn_h),
    'swap': pygame.Rect(20, HEIGHT - 40, btn_w, btn_h),
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
            nn = SimpleNN(n_inputs=2, n_hidden=3, n_outputs=2)

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            # check UI buttons first
            if buttons['pause_adapt'].collidepoint(mx, my):
                adapt_enabled = not adapt_enabled
            elif buttons['add_left'].collidepoint(mx, my):
                input_offsets[0] += const_step
            elif buttons['add_right'].collidepoint(mx, my):
                input_offsets[1] += const_step
            elif buttons['swap'].collidepoint(mx, my):
                swap_inputs = not swap_inputs
            else:
                # otherwise set the target position to mouse
                target[0] = mx
                target[1] = my
        

    # base inputs are normalized target coordinates, plus any user offsets
    nn_inputs = [
        target[0] / WIDTH + input_offsets[0],
        target[1] / HEIGHT + input_offsets[1]
    ]

    # optionally swap both inputs
    if swap_inputs:
        nn_inputs = [nn_inputs[1], nn_inputs[0]]

    # clamp inputs to [0,1]
    nn_inputs = [float(np.clip(i, 0.0, 1.0)) for i in nn_inputs]
    nn_outputs = nn.forward(nn_inputs)

    # clamp outputs to 0, 1
    nn_outputs = [np.clip(o, 0, 1) for o in nn_outputs]

    point[0] = WIDTH * nn_outputs[0]
    point[1] = HEIGHT * nn_outputs[1]

    error = nn.get_error(point, target)
    # every iteration_rate frames, adjust weights (if adaptation enabled)
    if frame_count % iteration_rate == 0:
        if adapt_enabled:
            nn.homeostatic_adjustment(error)
    draw_scene(screen, point, target)
    
    render.draw_nn(screen, nn, (500, 50))
    render.draw_text_info(screen, frame_count, error, nn_outputs)

    # draw UI buttons
    adapt_label = "Adaptation: ON" if adapt_enabled else "Adaptation: OFF"
    render.draw_button(screen, buttons['pause_adapt'], adapt_label, bg=(60,120,60) if adapt_enabled else (120,60,60))
    render.draw_button(screen, buttons['add_left'], f"Add +{const_step:.2f} to Left")
    render.draw_button(screen, buttons['add_right'], f"Add +{const_step:.2f} to Right")
    swap_label = "Swap Inputs: ON" if swap_inputs else "Swap Inputs: OFF"
    render.draw_button(screen, buttons['swap'], swap_label, bg=(80,80,100) if swap_inputs else (60,60,60))

    # draw offset status
    ofs_text = f"Offsets L={input_offsets[0]:.2f}  R={input_offsets[1]:.2f}"
    ofs_surf = render.FONT_SMALL.render(ofs_text, True, (240,240,240))
    screen.blit(ofs_surf, (20 + btn_w + 10, HEIGHT - 90))
    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()
sys.exit()