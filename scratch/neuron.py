class Neuron:
    def __init__(self, learning_rate=.01):
        self.learning_rate = learning_rate
        self.weight = 1
        self.bias = 0

    @property
    def parameters(self):
        return self.weight, self.bias

    @property
    def formula(self):
        return f"{self.weight:.4f}X + {self.bias:.4f}"

    def train(self, X, y, epochs=200):
        for epoch in range(epochs):
            for idx in range(len(X)):
                a = self.predict(X[idx])
                error = y[idx] - a

                self.weight += (self.learning_rate * error * X[idx])
                self.bias += (self.learning_rate * error)

    def predict(self, x):
        return self.weight*x + self.bias


X = [2, 3, 4, 11, 12, 15]
y = [5, 7, 9, 23, 25, 31]

neuron = Neuron()
neuron.train(X, y, epochs=1000)
print("The Formula: ", neuron.formula)
print("Predict(5): ", neuron.predict(5))
