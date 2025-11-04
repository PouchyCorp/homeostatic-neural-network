import pygame
import sys
import numpy as np
import random
import math

class Neuron:
    def __init__(self, n_inputs, mutation_step=0.005, improvement_tolerance=1.0):
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_inputs)]
        self.bias = random.uniform(-0.5, 0.5)
        self.output = 0.0
        # when True, this neuron will not perform adaptations
        self.blocked = False
        
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

    def adapt(self, error, target_threshold=0.1):
        """Ashby-style reversible homeostasis."""
        # do not adapt if neuron is blocked
        if getattr(self, 'blocked', False):
            return
        # first iteration: no previous error to compare
        if error < target_threshold:
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

nn = SimpleNN(n_inputs=2, n_hidden=2, n_outputs=2)

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
            nn = SimpleNN(n_inputs=2, n_hidden=3, n_outputs=2)

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