class Neuron:
    def __init__(self, id, layer, activation="sigmoid"):
        self.id = id
        self.layer = layer
        self.inbound_connections = []
        self.outbound_connections = []
        self.value = 0.0
        self.activation = activation
        self.input_sum = 0.0  # Store input sum for backprop
        self.bias = 0.0       # Add bias

    def add_inbound_connection(self, connection):
        self.inbound_connections.append(connection)

    def add_outbound_connection(self, connection):
        self.outbound_connections.append(connection)

    def activate(self):
        self.input_sum = sum(conn.weight * conn.from_neuron.value for conn in self.inbound_connections) + self.bias
        if self.activation == "sigmoid":
            self.value = self.sigmoid(self.input_sum)
        elif self.activation == "relu":
            self.value = max(0, self.input_sum)
        elif self.activation == "tanh":
            import math
            self.value = math.tanh(self.input_sum)
        elif self.activation == "linear":
            self.value = self.input_sum
        else:
            raise ValueError("Unknown activation")

    @staticmethod
    def sigmoid(x):
        import math
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = Neuron.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu_derivative(x):
        return 1 if x > 0 else 0

    @staticmethod
    def tanh_derivative(x):
        import math
        t = math.tanh(x)
        return 1 - t**2

    def __repr__(self):
        return f"Neuron(id={self.id}, layer={self.layer}, value={self.value}, bias={self.bias})"

class Connection:
    def __init__(self, from_neuron, to_neuron, weight):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight
        from_neuron.add_outbound_connection(self)
        to_neuron.add_inbound_connection(self)

    def __repr__(self):
        return f"Connection(from={self.from_neuron.id}, to={self.to_neuron.id}, weight={self.weight})"

# Example usage:
if __name__ == "__main__":
    import random

    n1 = Neuron(id=1, layer=0, activation="linear")
    hidden = [Neuron(id=2+i, layer=1, activation="relu") for i in range(8)]
    n_out = Neuron(id=20, layer=2, activation="linear")

    # Initialize biases
    for h in hidden:
        h.bias = random.uniform(-1, 1)
    n_out.bias = random.uniform(-1, 1)

    # Connections from input to hidden
    connections = []
    for h in hidden:
        connections.append(Connection(from_neuron=n1, to_neuron=h, weight=random.uniform(-1, 1)))
    # Connections from hidden to output
    for h in hidden:
        connections.append(Connection(from_neuron=h, to_neuron=n_out, weight=random.uniform(-1, 1)))

    training_data = [(x, x**2 + 3*x + 2) for x in range(-10, 11)]
    for epoch in range(3000):
        for x, y in training_data:
            n1.value = x
            for h in hidden:
                h.activate()
            n_out.activate()
            # Backpropagation
            error = n_out.value - y
            d_out = error
            # Update hidden->output weights and output bias
            for conn in n_out.inbound_connections:
                conn.weight -= 0.05 * d_out * conn.from_neuron.value
            n_out.bias -= 0.05 * d_out
            # Hidden layer gradients and updates
            for h in hidden:
                d_h = d_out * h.outbound_connections[0].weight * Neuron.relu_derivative(h.input_sum)
                for conn in h.inbound_connections:
                    conn.weight -= 0.05 * d_h * conn.from_neuron.value
                h.bias -= 0.05 * d_h
        if epoch % 500 == 0:
            print(f"Epoch {epoch}")

    check_data = [(x, x**2 + 3*x + 2) for x in range(1, 20)]
    for x, y in check_data:
        n1.value = x
        for h in hidden:
            h.activate()
        n_out.activate()
        print(f"Check Input: {x}, Predicted: {n_out.value:.2f}, Actual: {y}")