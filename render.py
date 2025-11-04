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
        radius = 16
        col = (200,200,200) if not getattr(neuron, 'blocked', False) else (140,80,80)
        pygame.draw.circle(surface, col, (x,y), radius)
        # draw blocked marker
        if getattr(neuron, 'blocked', False):
            pygame.draw.circle(surface, (255,50,50), (x,y), radius-6, width=2)
        # draw bias number
        bias_text = FONT_SMALL.render(f"{neuron.bias:.2f}", True, (40,40,40))
        surface.blit(bias_text, (x - bias_text.get_width()//2, y - bias_text.get_height()//2))
        
    # output layer
    output_pos = []
    for i, neuron in enumerate(nn.output.neurons):
        x = x0 + layer_spacing
        y = y0 + i * neuron_spacing + (len(hidden_pos)-len(nn.output.neurons))*neuron_spacing/2
        output_pos.append((x,y))
        radius = 20
        col = (200,200,200) if not getattr(neuron, 'blocked', False) else (140,80,80)
        pygame.draw.circle(surface, col, (x,y), radius)
        # draw blocked marker
        if getattr(neuron, 'blocked', False):
            pygame.draw.circle(surface, (255,50,50), (x,y), radius-6, width=2)
        # draw bias number
        bias_text = FONT_SMALL.render(f"{neuron.bias:.2f}", True, (240,240,240))
        surface.blit(bias_text, (x - bias_text.get_width()//2, y - bias_text.get_height()//2))
        # draw output value on the side
        out_text = FONT_SMALL.render(f"{getattr(neuron, 'output', 0.0):.2f}", True, (0,255,0))
        surface.blit(out_text, (x + 30, y - out_text.get_height()//2))

    # draw weights (hidden -> output)
    
    for i_h, h_neu in enumerate(nn.hidden.neurons):
        hx, hy = hidden_pos[i_h]
        for i_o, o_neu in enumerate(nn.output.neurons):
            ox, oy = output_pos[i_o]
            w = o_neu.weights[i_h]
            col = map_color_for_weight(w)
            width = max(1, int(1 + abs(w) * 3))
            pygame.draw.line(surface, col, (hx+8,hy), (ox-8,oy), width)
    # input nodes are not shown — this visualization focuses on hidden/output layers

    # return positions for external click handling
    return {
        'hidden_pos': hidden_pos,
        'output_pos': output_pos,
        'hidden_radius': 16,
        'output_radius': 20,
    }

def get_nn_positions(nn, origin, layer_spacing=200, neuron_spacing=40):
    """Compute neuron positions without drawing. Returns same structure as draw_nn's return."""
    x0, y0 = origin
    hidden_pos = []
    for i, _ in enumerate(nn.hidden.neurons):
        x = x0 + 50
        y = y0 + i * neuron_spacing
        hidden_pos.append((x, y))
    output_pos = []
    for i, _ in enumerate(nn.output.neurons):
        x = x0 + layer_spacing
        y = y0 + i * neuron_spacing + (len(hidden_pos)-len(nn.output.neurons))*neuron_spacing/2
        output_pos.append((x, y))
    return {
        'hidden_pos': hidden_pos,
        'output_pos': output_pos,
        'hidden_radius': 16,
        'output_radius': 20,
    }

def draw_text_info(surface, frame_count, error, output, target, nn_outputs, normalized_target):
    lines = [
        f"Frame: {frame_count}",
        f"Outputs: [{output[0]:.4f}, {output[1]:.4f}]",
        f"Target: [{target[0]:.4f}, {target[1]:.4f}]",
        f"Error: {error:.4f}",
        f"NN Outputs: [{nn_outputs[0]:.4f}, {nn_outputs[1]:.4f}]",
        f"Normalized Target: [{normalized_target[0]:.4f}, {normalized_target[1]:.4f}]",
    ]
    for i, line in enumerate(lines):
        text_surf = FONT_SMALL.render(line, True, (240,240,240))
        surface.blit(text_surf, (10, 10 + i * 16))