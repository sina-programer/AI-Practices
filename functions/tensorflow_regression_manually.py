from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np


class LinearRegression:
    def __init__(self):
        self.weight = tf.Variable(np.random.randn())
        self.bias = tf.Variable(np.random.randn())

    @property
    def params(self):
        return [self.weight, self.bias]

    @property
    def formula(self):
        return f"Y = {self.weight.numpy():.2f}X + {self.bias.numpy():.2f}"

    def predict(self, X):
        return self.weight*X + self.bias

    def fit(self, X, y, epochs=100, lr=0.01):
        for epoch in range(1, epochs+1):
            self.train_step(X, y, lr=lr)
            print(f"Epoch {epoch}:  Loss={self.loss(y, self.predict(X))}")

    def train_step(self, X, y, lr=.01):
        with tf.GradientTape() as t:
            current_loss = self.loss(y, self.predict(X))
            weight, bias = t.gradient(current_loss, self.params)

        self.weight.assign_sub(weight * lr)  # self.weight -= weight*lr
        self.bias.assign_sub(bias * lr)

    def loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))



X = np.linspace(0, 10, 150)
y = 2*X + 1 + np.random.randn(len(X))

model = LinearRegression()
model.fit(X, y)

plt.scatter(X, y, s=7)
plt.plot(X, model.predict(X), color='red', lw=1, linestyle='--')
plt.title(model.formula)
plt.show()
