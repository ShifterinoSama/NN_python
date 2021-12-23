import numpy as np

class Perceptron():
    
    def __init__(self, number_of_inputs=3, learning_rate=0.01) -> None:
        self.weights = np.empty((number_of_inputs))
        self.learning_rate = learning_rate
        # Náhodně inicializuje váhu pro každý vstup
        for i in range(len(self.weights)):
            self.weights[i] = np.random.uniform(-1,1)
    
    def activation(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    def guess(self, inputs):
        sum = 0
        for i in range(len(self.weights)):
            sum += inputs[i] * self.weights[i]
        return self.activation(sum)
    
    # Optimalizovat váhy
    def train(self, inputs, target):
        guess = self.guess(inputs)
        error = target - guess
        for i in range(len(self.weights)):
            self.weights[i] += error * inputs[i] * self.learning_rate
    
    def guess_Y(self, x):
        w0 = self.weights[0]
        w1 = self.weights[1]
        w2 = self.weights[2]
        return -(w2/w1)-(w0/w1)*x