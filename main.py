import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def damage(X, percent, seed = 1):
  rgen = np.random.RandomState(seed)
  result = np.array(X)
  count = int(X.shape[1]*percent/100)
  for indeks_example in range(len(X)):
    order = np.sort(rgen.choice(X.shape[1], count, replace=False))
    for indeks_pixel in order:
      result[indeks_example][indeks_pixel] *= -1

  return result
class Perceptron(object):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

class SLP:
    def __init__(self, eta=0.05, n_iter=10, random_state=1):
        self.perceptrons = []
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        self.perceptrons = []
        for _ in range(y.shape[1]):
            ppn = Perceptron(eta=self.eta, n_iter=self.n_iter, random_state = self.random_state)
            ppn.fit(X, y[:, _])
            self.perceptrons.append(ppn)
        return self

    def predict(self, X):
        predictions = []
        for ppn in self.perceptrons:
            predictions.append(ppn.predict(X))
        return np.array(predictions).T

    def misclassified(self, X, y):
        return (self.predict(X) != y).sum()
    
    def show(self, X):
        for xi in X:
            pixels = xi.reshape(7,5)
            plt.imshow(pixels, cmap='binary')
            plt.show()
df = pd.read_csv("dane.txt", header=None)
set = [7, 8, 11, 13, 16, 19, 21, 22, 24, 25]
x = df.iloc[set,:35].values
y = df.iloc[0:10, 35:45].values
net = SLP()
net.show(x)
net.fit(x,y)
print(net.predict(x))
print(net.misclassified(x,y))
damage5 = damage(x,5)
damage15 = damage(x,15)
damage40 = damage(x,40)
print(net.predict(damage5))
print(net.misclassified(damage5,y))
print(net.predict(damage15))
print(net.misclassified(damage15,y))
print(net.predict(damage40))
print(net.misclassified(damage40,y))

