import random
import math

class Neuron:
    def __init__(self, n_inputs, mutation_step=0.005, improvement_tolerance=1.0, zero_init=False, error_scaling=False):
        if zero_init:
            self.weights = [0.0 for _ in range(n_inputs)]
            self.bias = 0.0
        else:
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
        self.error_scaling = error_scaling

    def activate(self):
        z = sum(w for w in self.weights) + self.bias
        self.output = math.tanh(z)
        return self.output

    def _mutate(self, error, minmaxerror : tuple):
        """Store a reversible mutation."""
        idx = random.randrange(len(self.weights) + 1)
        # Ensure we mutate in the current persistent direction (sign)
        sign = self.mutation_direction if self.mutation_direction in (1, -1) else random.choice([-1, 1])
        # small random magnitude, but fixed sign
        
        # the larger the error, the larger the mutation step
        if self.error_scaling:
            error_offset = min(max((error - minmaxerror[0]) / (minmaxerror[1] - minmaxerror[0] + 1e-9), 0), 1) * 0.1
        else:
            error_offset = 0.0
        delta = sign * random.uniform(0, self.mutation_step + error_offset)

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

    def adapt(self, error, minmaxerror, tolerance=0.1):
        """Ashby-style reversible homeostasis."""
        # do not adapt if neuron is blocked
        if getattr(self, 'blocked', False):
            return
        # first iteration: no previous error to compare
        if error < tolerance:
            return  # no adaptation needed for low error

        if self.prev_error is None:
            self.prev_error = error
            self._mutate(error, minmaxerror)
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
        self._mutate(error , minmaxerror)

        self.prev_error = error


class Layer:
    def __init__(self, n_neurons, n_inputs, zero_init=False, error_scaling=False):
        self.neurons = [Neuron(n_inputs, zero_init=zero_init, error_scaling=error_scaling) for _ in range(n_neurons)]

    def forward(self):
        return [n.activate() for n in self.neurons]

class Homeostat:
    def __init__(self, n_hidden, n_outputs, zero_init=False, error_scaling = False):
        self.hidden = Layer(n_hidden, 2 , zero_init=zero_init, error_scaling=error_scaling)
        self.output = Layer(n_outputs, n_hidden, zero_init=zero_init, error_scaling=error_scaling)

    def forward(self):
        h = self.hidden.forward()
        return self.output.forward()

    def homeostatic_adjustment(self, error, minmaxerror =(0.0, 100.0), tolerance=0.1):
        for neuron in self.hidden.neurons + self.output.neurons:
            neuron.adapt(error, minmaxerror, tolerance)

    def get_error(self, point, target):
        return math.hypot(point[0]-target[0], point[1]-target[1])