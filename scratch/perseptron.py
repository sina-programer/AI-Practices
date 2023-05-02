import numpy as np
import pandas as pd


class Perseptron:
    def __init__(self, input_dim, learning_rate=.1, bias=1):
        self._is_trained = False
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.bias = bias
        self.bias_weight = np.random.random()
        self.weights = np.random.random(size=self.input_dim)

    def train(self, X, y, epochs=200):
        for epoch in range(1, epochs+1):
            print(f"    Epoch: {epoch:<4} ".center(50, '-'))
            total_error = 0

            for idx, row in X.iterrows():
                result = self._predict([row])[0]
                error = y.iloc[idx] - result
                total_error += abs(error)

                self.bias_weight += (self.learning_rate * error * self.bias)
                for i in range(self.input_dim):
                    self.weights[i] += (self.learning_rate * error * row.iloc[i])

            print("Error: ", total_error)

        self._is_trained = True
        print('-'*50)

    def _predict(self, X):
        y = []
        for row in X:
            coefficients = np.sum(row * self.weights) + (self.bias * self.bias_weight)
            coefficients = Perseptron._activation_function(coefficients)
            y.append(coefficients)

        return y

    def predict(self, X):
        if self._is_trained:
            return self._predict(X.values)

        raise PermissionError('First train the model')

    @staticmethod
    def _activation_function(number):
        if number > 0:
            return 1
        return 0


df = pd.DataFrame({
    'Distance': [10, 5, 0, 4, 3, 7, 6, 2, 8, 1],
    'Rate': [    10, 5, 0, 1, 4, 5, 6, 2, 9, 1],
    'Prize': [    1, 1, 0, 0, 0, 1, 1, 0, 1, 0]
})
print(df)

X = df.iloc[:, :-1]
y = df['Prize']

X_test = pd.DataFrame({
    'Distance': [.5, 7, 0, 6, 9],
    'Rate': [     5, 6, 1, 8, 2],
    # 'Prize': [  0, 1, 0, 1, ?]
})

perseptron = Perseptron(input_dim=2)
perseptron.train(X, y, epochs=50)
print("Weights: ", perseptron.weights)
print("Bias / Bias-Weight:  ", perseptron.bias, ' / ', perseptron.bias_weight)
X_test['Prize'] = perseptron.predict(X_test)
print(X_test)
