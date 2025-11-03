# pendulum_nn_homeostat.py
import pygame
import random
import math
from collections import deque
import time

# -------------------------
# Pendulum with moving base
# -------------------------
class MovingBasePendulum:
    def __init__(self, l=0.6, g=9.81, dt=0.02):
        self.l = l
        self.g = g
        self.dt = dt
        self.reset()

    def reset(self, theta=None, theta_dot=0.0, x_base=0.0, x_base_dot=0.0):
        # theta: 0 = vertical down. We'll present near upright by using pi ~ downward
        if theta is None:
            # start near upright but slightly perturbed
            self.theta = math.pi - 0.2 * random.uniform(-1, 1)
        else:
            self.theta = theta
        self.theta_dot = theta_dot
        self.x_base = x_base
        self.x_base_dot = x_base_dot
        self.x_base_ddot = 0.0

    def step(self, base_acc):
        """Advance one time step given base horizontal acceleration."""
        self.x_base_ddot = base_acc
        # integrate base motion
        self.x_base_dot += self.x_base_ddot * self.dt
        self.x_base += self.x_base_dot * self.dt

        # angular acceleration for pendulum with accelerating base:
        # Correct EOM (theta measured from vertical down):
        # theta_ddot = (g / l) * sin(theta) - (base_acc / l) * cos(theta)
        # Note: theta=0 down, theta=pi up. We'll treat upright goal as theta ~ pi
        theta_ddot = (self.g / self.l) * math.sin(self.theta) - (base_acc / self.l) * math.cos(self.theta)

        # integrate angular state
        self.theta_dot += theta_ddot * self.dt
        self.theta += self.theta_dot * self.dt

        # wrap theta to [-pi, pi] for stability in display/metrics
        self.theta = ((self.theta + math.pi) % (2 * math.pi)) - math.pi

        return self.get_state()

    def get_state(self):
        return (self.theta, self.theta_dot, self.x_base, self.x_base_dot)

    def tip_position(self):
        # return (x_tip, y_tip) relative coords: pivot at (x_base, 0)
        x_tip = self.x_base + self.l * math.sin(self.theta)
        y_tip = -self.l * math.cos(self.theta)
        return x_tip, y_tip

# -------------------------
# Simple class-based NN
# -------------------------
class Neuron:
    def __init__(self, n_inputs):
        self.weights = [random.uniform(-1.0, 1.0) for _ in range(n_inputs)]
        self.bias = random.uniform(-0.5, 0.5)
        self.output = 0.0

    def activate(self, inputs):
        s = self.bias
        for w, inp in zip(self.weights, inputs):
            s += w * inp
        # tanh activation
        self.output = math.tanh(s)
        return self.output

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
        out = self.output.forward(h)
        return out, h  # return both for adaptation heuristics

    def get_all_weights(self):
        # return a list of tuples (layer_name, neuron_idx, weight_idx, reference)
        items = []
        for layer_name, layer in (("hidden", self.hidden), ("output", self.output)):
            for ni, neuron in enumerate(layer.neurons):
                for wi in range(len(neuron.weights)):
                    items.append((layer_name, ni, wi, neuron))
        return items

    def mutate_one(self, scale=0.3):
        """Pick a weight or bias randomly and mutate it. Return a small mutation descriptor to revert if needed."""
        layer_choice = random.choice(["hidden", "output"])
        layer = self.hidden if layer_choice == "hidden" else self.output
        ni = random.randrange(len(layer.neurons))
        neuron = layer.neurons[ni]
        mutate_type = random.choice(["weight", "bias"])
        if mutate_type == "weight":
            wi = random.randrange(len(neuron.weights))
            old = neuron.weights[wi]
            delta = random.uniform(-scale, scale)
            neuron.weights[wi] += delta
            return ("weight", layer_choice, ni, wi, old)
        else:
            old = neuron.bias
            delta = random.uniform(-scale, scale)
            neuron.bias += delta
            return ("bias", layer_choice, ni, None, old)

    def revert(self, desc):
        typ, layer_choice, ni, wi, old = desc
        layer = self.hidden if layer_choice == "hidden" else self.output
        neuron = layer.neurons[ni]
        if typ == "weight":
            neuron.weights[wi] = old
        else:
            neuron.bias = old

# -------------------------
# Homeostatic controller manager
# -------------------------
class Homeostat:
    def __init__(self, nn, eval_window=20, instability_threshold=0.25, monitor_frames=200):
        self.nn = nn
        self.eval_window = eval_window
        self.instability_threshold = instability_threshold
        self.monitor_frames = monitor_frames

        self.error_history = deque(maxlen=200)
        self.E_running = 1e6

        # candidate mutation bookkeeping
        self.candidate = None
        self.candidate_start_time = None
        self.pre_candidate_error = None
        self.monitor_count = 0
        self.post_accum_error = 0.0

    def measure_error(self, theta, theta_dot):
        # goal: upright -> theta ~ pi (we'll measure around pi)
        # normalize angular error to [-pi, pi]
        angle_err = ((theta - math.pi + math.pi) % (2 * math.pi)) - math.pi
        e = (angle_err)**2 + 0.05 * (theta_dot**2)
        self.error_history.append(e)
        self.E_running = sum(self.error_history)/len(self.error_history)

        # punish for being near edges of base position (if pend is available)
        try:
            if abs(pend.x_base) >= 3:
                e += 5.0  # large penalty for being at edges
        except NameError:
            pass

        # punish a lot for high angular velocity
        e += 0.5 * (theta_dot**2)
        return e

    def maybe_try_mutation(self):
        # If system unstable, and we're not already testing a candidate, create one
        if self.E_running > self.instability_threshold and self.candidate is None:
            desc = self.nn.mutate_one(scale=0.5)  # bigger exploratory jump
            self.candidate = desc
            self.candidate_start_time = time.time()
            self.pre_candidate_error = self.E_running
            self.monitor_count = 0
            self.post_accum_error = 0.0

    def evaluate_candidate(self, current_error):
        if self.candidate is None:
            return
        self.monitor_count += 1
        self.post_accum_error += current_error
        if self.monitor_count >= self.monitor_frames:
            post_avg = self.post_accum_error / self.monitor_count
            # keep if post_avg < pre_candidate_error (improvement)
            if post_avg < self.pre_candidate_error:
                # keep mutation, maybe shrink future mutation scale (not implemented)
                pass  # mutation already applied
            else:
                # revert
                self.nn.revert(self.candidate)
            # clear candidate
            self.candidate = None
            self.pre_candidate_error = None
            self.monitor_count = 0
            self.post_accum_error = 0.0

# -------------------------
# GUI & main loop (pygame)
# -------------------------
def map_color_for_weight(w):
    # green for positive, red for negative, intensity by magnitude
    mag = min(abs(w), 1.5)
    
    if w >= 0:
        col =  (int(50 * mag + 50), int(200 * mag), 50)
    else:
        col =  (200, int(100 * (1 - mag/1.5)), int(70 * (1 - mag/1.5)))

    col = tuple(min(255, max(0, c)) for c in col)
    return col

def draw_nn(surface, nn, origin, layer_spacing=200, neuron_spacing=40):
    x0, y0 = origin
    # hidden layer
    hidden_pos = []
    for i, neuron in enumerate(nn.hidden.neurons):
        x = x0 + 50
        y = y0 + i * neuron_spacing
        hidden_pos.append((x, y))
        pygame.draw.circle(surface, (200,200,200), (x,y), 10)
    # output layer
    output_pos = []
    for i, neuron in enumerate(nn.output.neurons):
        x = x0 + layer_spacing
        y = y0 + i * neuron_spacing + (len(hidden_pos)-len(nn.output.neurons))*neuron_spacing/2
        output_pos.append((x,y))
        pygame.draw.circle(surface, (200,200,200), (x,y), 12)

    # draw weights (hidden -> output)
    for i_h, h_neu in enumerate(nn.hidden.neurons):
        hx, hy = hidden_pos[i_h]
        for i_o, o_neu in enumerate(nn.output.neurons):
            ox, oy = output_pos[i_o]
            w = o_neu.weights[i_h]
            col = map_color_for_weight(w)
            width = max(1, int(1 + abs(w) * 3))
            pygame.draw.line(surface, col, (hx+8,hy), (ox-8,oy), width)

    # draw input->hidden weights as smaller lines with a background label
    # we won't draw individual input nodes, just small glyphs
    # label inputs
    input_labels = ["sinθ","cosθ","θ̇","base_v"]
    for i_in, label in enumerate(input_labels):
        ix = x0 - 80
        iy = y0 + i_in * 18
        color = (220,220,220)
        surface.blit(FONT_SMALL.render(label, True, (240,240,240)), (ix-10, iy-8))
        # draw small lines to each hidden neuron scaled by each neuron's weight
        for i_h, h_neu in enumerate(nn.hidden.neurons):
            w = h_neu.weights[i_in]
            col = map_color_for_weight(w)
            hx, hy = hidden_pos[i_h]
            pygame.draw.line(surface, col, (ix+40, iy+6), (hx-8, hy), max(1, int(1 + abs(w)*2)))

def draw_angle_graph(surface, angle_history, rect):
    pygame.draw.rect(surface, (20,20,20), rect)
    if len(angle_history) < 2:
        return
    w = rect[2]; h = rect[3]; x0 = rect[0]; y0 = rect[1]
    maxlen = 200
    vals = list(angle_history)[-maxlen:]
    # map angles to [-1,1] by scaling
    # show upright=0 -> center of graph
    maxval = 1.5  # radians for scale
    pts = []
    for i, v in enumerate(vals):
        xx = x0 + int(i * (w / maxlen))
        yy = y0 + h//2 - int((v / maxval) * (h//2))
        pts.append((xx, yy))
    if len(pts) > 1:
        pygame.draw.lines(surface, (100,200,250), False, pts, 2)
    # draw center line
    pygame.draw.line(surface, (80,80,80), (x0, y0+h//2), (x0+w, y0+h//2), 1)

# -------------------------
# Main runnable demo
# -------------------------
if __name__ == "__main__":
    pygame.init()
    WIDTH, HEIGHT = 1000, 600
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pendulum + Homeostatic NN (trial/error)")

    # fonts
    global FONT_SMALL
    FONT_SMALL = pygame.font.SysFont("Consolas", 14)
    font = pygame.font.SysFont("Consolas", 18)

    clock = pygame.time.Clock()

    # simulation objects
    pend = MovingBasePendulum(l=0.8, dt=0.02)
    nn = SimpleNN(n_inputs=4, n_hidden=6, n_outputs=1)
    homeo = Homeostat(nn, eval_window=20, instability_threshold=0.3, monitor_frames=30)

    angle_history = deque(maxlen=300)
    running = True

    # GUI positions
    pivot_screen_x = WIDTH//3
    pivot_screen_y = HEIGHT//3

    perturb_impulse = 2.0  # instantaneous base velocity change

    while running:
        # ----- event handling -----
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    # apply a perturbation: add a sudden velocity to the base
                    pend.x_base_dot += perturb_impulse * (0.8 + random.random()*0.4)
                elif ev.key == pygame.K_r:
                    pend.reset()
                    angle_history.clear()
                elif ev.key == pygame.K_p:
                    # optional: print current weights
                    print("Hidden weights:")
                    for i, n in enumerate(nn.hidden.neurons):
                        print(i, ["{:+.2f}".format(w) for w in n.weights], "bias", "{:+.2f}".format(n.bias))

        # ----- controller: compute inputs and get NN output -----
        theta, theta_dot, x_base, x_base_dot = pend.get_state()
        # normalize inputs: sin, cos, angular velocity, base velocity (clamped)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        inp = [sin_t, cos_t, max(-5.0, min(5.0, theta_dot)), max(-5.0, min(5.0, x_base_dot))]
        out, hidden_act = nn.forward(inp)
        # output is tanh-like since neuron uses tanh => value in (-1,1)
        base_acc_cmd = out[0] * 9.0  # scale to a reasonable accel (m/s^2)

        # step physics
        pend.step(base_acc_cmd)

        # Constrain the base so it stays on screen.
        # The rendering maps world x to screen via: px = pivot_screen_x + int(pend.x_base * 120)
        # Compute allowable world x range so px stays within screen bounds (account for base width).
        scale = 120
        base_w = 60
        min_px = base_w // 2
        max_px = WIDTH - base_w // 2
        min_x_base = (min_px - pivot_screen_x) / scale
        max_x_base = (max_px - pivot_screen_x) / scale
        if pend.x_base < min_x_base:
            pend.x_base = min_x_base
            # if velocity points further outwards (negative), zero it; allow positive velocity to move back in
            if pend.x_base_dot < 0.0:
                pend.x_base_dot = 0.0
        elif pend.x_base > max_x_base:
            pend.x_base = max_x_base
            # if velocity points further outwards (positive), zero it; allow negative velocity to move back in
            if pend.x_base_dot > 0.0:
                pend.x_base_dot = 0.0

        # compute error and feed homeostat
        err = homeo.measure_error(theta, theta_dot)
        homeo.maybe_try_mutation()
        homeo.evaluate_candidate(err)

        # logging for graph: store angle error relative to upright (pi) so graph centers on upright
        angle_err = ((theta - math.pi + math.pi) % (2 * math.pi)) - math.pi
        angle_history.append(angle_err)

        # ----- rendering -----
        screen.fill((12, 12, 20))

        # draw pendulum base (converted to screen coords)
        px = pivot_screen_x + int(pend.x_base * 120)  # scale base x to screen
        py = pivot_screen_y
        base_w, base_h = 60, 12
        pygame.draw.rect(screen, (160,160,160), (px-base_w//2, py-base_h//2, base_w, base_h))
        # draw pivot circle
        pygame.draw.circle(screen, (220,220,220), (px, py), 6)

        # draw pendulum rod and tip
        x_tip_rel, y_tip_rel = pend.tip_position()
        tip_x = pivot_screen_x + int(x_tip_rel * 120)
        tip_y = pivot_screen_y + int(y_tip_rel * 120)
        pygame.draw.line(screen, (240,240,240), (px,py), (tip_x, tip_y), 4)
        pygame.draw.circle(screen, (250,120,80), (tip_x, tip_y), 12)

        # text info
        screen.blit(font.render(f"theta: {theta:.2f} rad", True, (220,220,220)), (10,10))
        screen.blit(font.render(f"theta_dot: {theta_dot:.2f}", True, (220,220,220)), (10,34))
        screen.blit(font.render(f"base_x: {pend.x_base:.2f}", True, (220,220,220)), (10,58))
        screen.blit(font.render(f"error(E): {homeo.E_running:.3f}", True, (220,200,120)), (10,82))
        screen.blit(font.render(f"base_acc_cmd: {base_acc_cmd:.2f} m/s²", True, (220,220,220)), (10,106))

        # draw NN (top-right)
        draw_nn(screen, nn, origin=(WIDTH-420, 120), layer_spacing=140, neuron_spacing=26)

        # draw angle graph (bottom-right)
        draw_angle_graph(screen, angle_history, rect=(WIDTH-420, HEIGHT-220, 380, 180))

        # indicate if candidate mutation active
        if homeo.candidate is not None:
            screen.blit(font.render("Adapting... trying mutation", True, (240,120,120)), (WIDTH-420, 80))

        pygame.display.flip()
        clock.tick(50)  # ~50 FPS

    pygame.quit()
