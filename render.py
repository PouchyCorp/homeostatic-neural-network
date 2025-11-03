import pygame
pygame.font.init()
FONT_SMALL = pygame.font.SysFont("Arial", 14)

def draw_button(surface, rect, text, bg=(60,60,60), fg=(240,240,240)):
    """Draw a simple rectangular button with centered text."""
    pygame.draw.rect(surface, bg, rect, border_radius=4)
    # border
    pygame.draw.rect(surface, (100,100,100), rect, width=1, border_radius=4)
    text_surf = FONT_SMALL.render(text, True, fg)
    tx = rect.x + (rect.width - text_surf.get_width()) // 2
    ty = rect.y + (rect.height - text_surf.get_height()) // 2
    surface.blit(text_surf, (tx, ty))

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
        # draw bias number
        bias_text = FONT_SMALL.render(f"{neuron.bias:.2f}", True, (240,240,240))
        surface.blit(bias_text, (x - bias_text.get_width()//2, y))
    # output layer
    output_pos = []
    for i, neuron in enumerate(nn.output.neurons):
        x = x0 + layer_spacing
        y = y0 + i * neuron_spacing + (len(hidden_pos)-len(nn.output.neurons))*neuron_spacing/2
        output_pos.append((x,y))
        pygame.draw.circle(surface, (200,200,200), (x,y), 12)
        # draw bias number
        bias_text = FONT_SMALL.render(f"{neuron.bias:.2f}", True, (240,240,240))
        surface.blit(bias_text, (x - bias_text.get_width()//2, y))

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
    input_labels = ["Left Sensor", "Right Sensor"]
    for i_in, label in enumerate(input_labels):
        ix = x0 - 80
        iy = y0 + i_in * neuron_spacing
        color = (220,220,220)
        surface.blit(FONT_SMALL.render(label, True, (240,240,240)), (ix-10, iy-8))
        # draw small lines to each hidden neuron scaled by each neuron's weight
        for i_h, h_neu in enumerate(nn.hidden.neurons):
            w = h_neu.weights[i_in]
            col = map_color_for_weight(w)
            hx, hy = hidden_pos[i_h]
            pygame.draw.line(surface, col, (ix+40, iy+6), (hx-8, hy), max(1, int(1 + abs(w)*2)))

def draw_text_info(surface, frame_count, error, input, output):
    lines = [
        f"Frame: {frame_count}",
        f"Inputs: [{input[0]:.4f}, {input[1]:.4f}]",
        f"Outputs: [{output[0]:.4f}, {output[1]:.4f}]",
        f"Error: {error:.4f}",
    ]
    for i, line in enumerate(lines):
        text_surf = FONT_SMALL.render(line, True, (240,240,240))
        surface.blit(text_surf, (10, 10 + i * 16))