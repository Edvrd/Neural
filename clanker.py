import random
import math

class Neuron:
    def __init__(self, id, layer, activation="sigmoid"):
        self.id = id
        self.layer = layer
        self.inbound_connections = []
        self.outbound_connections = []
        self.value = 0.0
        self.activation = activation
        self.input_sum = 0.0 
        self.bias = 0.0
        self.delta = 0.0

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
            self.value = math.tanh(self.input_sum)
        elif self.activation == "linear":
            self.value = self.input_sum
        else:
            raise ValueError("Unknown activation")

    def get_activation_derivative(self):
        if self.activation == "sigmoid":
            return self.sigmoid_derivative(self.input_sum)
        elif self.activation == "relu":
            return self.relu_derivative(self.input_sum)
        elif self.activation == "tanh":
            return self.tanh_derivative(self.input_sum)
        elif self.activation == "linear":
            return 1.0
        else:
            raise ValueError("Unknown activation")

    @staticmethod
    def sigmoid(x):
        x = max(-500, min(500, x))
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        s = Neuron.sigmoid(x)
        return s * (1 - s)

    @staticmethod
    def relu_derivative(x):
        return 1.0 if x > 0 else 0.0

    @staticmethod
    def tanh_derivative(x):
        t = math.tanh(x)
        return 1 - t**2

    def __repr__(self):
        return f"Neuron(id={self.id}, layer={self.layer}, value={self.value:.3f}, bias={self.bias:.3f})"


class Connection:
    def __init__(self, from_neuron, to_neuron, weight):
        self.from_neuron = from_neuron
        self.to_neuron = to_neuron
        self.weight = weight
        from_neuron.add_outbound_connection(self)
        to_neuron.add_inbound_connection(self)

    def __repr__(self):
        return f"Connection(from={self.from_neuron.id}, to={self.to_neuron.id}, weight={self.weight:.3f})"


def he_initialization(n_inputs):
    return random.gauss(0, math.sqrt(2.0 / n_inputs))


def normalize_data(data):
    x_values = [x for x, y in data]
    y_values = [y for x, y in data]
    
    x_mean = sum(x_values) / len(x_values)
    x_std = math.sqrt(sum((x - x_mean)**2 for x in x_values) / len(x_values))
    
    y_mean = sum(y_values) / len(y_values)
    y_std = math.sqrt(sum((y - y_mean)**2 for y in y_values) / len(y_values))
    
    normalized = [((x - x_mean) / x_std, (y - y_mean) / y_std) for x, y in data]
    
    return normalized, (x_mean, x_std), (y_mean, y_std)


def denormalize_output(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


if __name__ == "__main__":
    n1 = Neuron(id=1, layer=0, activation="linear")
    hidden = [Neuron(id=2+i, layer=1, activation="relu") for i in range(8)]
    n_out = Neuron(id=20, layer=2, activation="linear")

    for h in hidden:
        h.bias = random.uniform(-0.1, 0.1)
    n_out.bias = 0.0

    connections = []
    for h in hidden:
        weight = he_initialization(1)
        connections.append(Connection(from_neuron=n1, to_neuron=h, weight=weight))
    
    for h in hidden:
        weight = he_initialization(len(hidden))
        connections.append(Connection(from_neuron=h, to_neuron=n_out, weight=weight))

    training_data = [(x, x**2 + 3*x + 2) for x in range(-10, 11)]
    normalized_data, (x_mean, x_std), (y_mean, y_std) = normalize_data(training_data)
    
    learning_rate = 0.001
    
    print("Training neural network to approximate f(x) = x^2 + 3x + 2\n")
    
    for epoch in range(10000):
        total_error = 0
        for x_norm, y_norm in normalized_data:
            n1.value = x_norm
            for h in hidden:
                h.activate()
            n_out.activate()
            
            error = n_out.value - y_norm
            total_error += error ** 2
            
            n_out.delta = error * n_out.get_activation_derivative()
            
            for h in hidden:
                error_sum = sum(conn.weight * conn.to_neuron.delta for conn in h.outbound_connections)
                h.delta = error_sum * h.get_activation_derivative()
            
            for conn in n_out.inbound_connections:
                conn.weight -= learning_rate * n_out.delta * conn.from_neuron.value
            n_out.bias -= learning_rate * n_out.delta
            
            for h in hidden:
                for conn in h.inbound_connections:
                    conn.weight -= learning_rate * h.delta * conn.from_neuron.value
                h.bias -= learning_rate * h.delta
        
        if epoch % 500 == 0:
            mse = total_error / len(normalized_data)
            print(f"Epoch {epoch:4d} | MSE: {mse:.6f}")
    
    print("\n" + "="*60)
    print("Testing on new data:")
    print("="*60)
    
    check_data = [(x, x**2 + 3*x + 2) for x in range(-5, 16)]
    print(f"{'Input':>6} | {'Predicted':>10} | {'Actual':>10} | {'Error':>10}")
    print("-" * 60)
    
    for x, y_actual in check_data:
        x_norm = (x - x_mean) / x_std
        
        n1.value = x_norm
        for h in hidden:
            h.activate()
        n_out.activate()
        
        y_pred = denormalize_output(n_out.value, y_mean, y_std)
        error = abs(y_pred - y_actual)
        
        print(f"{x:6d} | {y_pred:10.2f} | {y_actual:10d} | {error:10.2f}")
    
    print("\n" + "="*60)
    print("Network Statistics:")
    print("="*60)
    print(f"Active hidden neurons: {sum(1 for h in hidden if any(abs(c.weight) > 0.01 for c in h.outbound_connections))}/{len(hidden)}")
    print(f"Total connections: {len(connections)}")
