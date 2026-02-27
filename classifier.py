import numpy as np

class NeuralNetwork:
    """
    A neural network with one hidden layer trained with backpropagation.
    Takes in a range of input features which describe the game state and
    predicts which direction Pacman should move. Uses momentum and L2
    regularisation to find optimal weights that generalise without overfitting.
    """
    def __init__(self, input_size, hidden_neurons, output_size, learning_rate, epochs, weight_decay=0.01, momentum=0.9):
        self.lr = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.momentum = momentum

        # Small beginning weights that are randomised to avoid having all weights the same and biases set to zero
        self.W1 = np.random.randn(input_size, hidden_neurons) * 0.01
        self.b1 = np.zeros((1, hidden_neurons))
        self.W2 = np.random.randn(hidden_neurons, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        # setting the velocities to zero for all weights and biases ready for momentum calculations
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X):
        # calculating hidden layer of neurons using ReLU activation for computational efficiency and to
        # mitigate the vanishing gradient problem
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)

        # computing output layer raw scores before softmax
        self.Z2 = np.dot(self.A1, self.W2) + self.b2

        # softmax calculation to convert raw scores into probabilities, subtracting max to avoid
        # massive numbers from large exponentials
        shifted = self.Z2 - np.max(self.Z2, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        # calculating the softmax score for each output neuron
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return self.probs

    def backward(self, X, y):
        # N is the number of training examples, used to average gradients
        N = X.shape[0]

        # one-hot encoding the target labels 0 - 3 ready to deduct from our probabilities
        y_onehot = np.zeros_like(self.probs)
        y_onehot[np.arange(N), y] = 1

        # calculating the responsibility of error loss for each output neuron to then update the second layer of weights
        dZ2 = self.probs - y_onehot
        dW2 = np.dot(self.A1.T, dZ2) / N + self.weight_decay * self.W2  # L2 regularisation to prevent overfitting
        db2 = np.sum(dZ2, axis=0, keepdims=True) / N

        # gradient of the loss with respect to Z1 (pre-ReLU scores), propagated back through W2
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = np.dot(X.T, dZ1) / N + self.weight_decay * self.W1
        db1 = np.sum(dZ1, axis=0, keepdims=True) / N

        # decayed previous velocity added with the current gradient to produce a new velocity,
        # helping the network build up speed in consistent directions and avoid local minima
        self.vW1 = self.momentum * self.vW1 + dW1
        self.vb1 = self.momentum * self.vb1 + db1
        self.vW2 = self.momentum * self.vW2 + dW2
        self.vb2 = self.momentum * self.vb2 + db2

        # step weights in the direction of the accumulated velocity
        self.W1 -= self.lr * self.vW1
        self.b1 -= self.lr * self.vb1
        self.W2 -= self.lr * self.vW2
        self.b2 -= self.lr * self.vb2

    def fit(self, X, y):
        for epoch in range(self.epochs):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        probs = self.forward(X)
        # return the index of the highest probability class
        return np.argmax(probs, axis=1)


class Classifier:
    # direction encoding used in good-moves.txt translated so that it is compatible with legal moves
    DIRECTION_MAP = {'North': 0, 'East': 1, 'South': 2, 'West': 3}

    def __init__(self):
        self.model = None

    def reset(self):
        self.model = None

    def fit(self, data, target):
        # convert training data to numpy arrays for matrix operations
        X = np.array(data)
        y = np.array(target)

        # building out the neural network with input size matching the feature vector length,
        # hidden size of 32 should find patterns in the data without overfitting
        self.model = NeuralNetwork(
            input_size=X.shape[1],
            hidden_neurons=32,
            output_size=4,
            learning_rate=0.01,
            epochs=500,
            weight_decay=0.01,
            momentum=0.9)

        self.model.fit(X, y)

    def predict(self, features, legal):
        # if the model is yet to be trained just pick a random direction
        if self.model is None:
            return np.random.randint(4)

        # reshape features into a single row matrix compatible with the input the neural network expects
        X = np.array(features).reshape(1, -1)
        prediction = self.model.predict(X)[0]

        # convert legal move strings to numeric equivalents
        legal_numbers = []
        for move in legal:
            if move == 'North':
                legal_numbers.append(0)
            elif move == 'East':
                legal_numbers.append(1)
            elif move == 'South':
                legal_numbers.append(2)
            elif move == 'West':
                legal_numbers.append(3)

        if prediction in legal_numbers:
            return prediction
