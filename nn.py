import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neural_Network():
    def __init__(self,input_nodes, hidden_nodes, output_nodes) -> None:
        self.input_nodes = input_nodes 
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.weights_input_hidden = np.empty((hidden_nodes, input_nodes))
        self.weights_hidden_output = np.empty((output_nodes, hidden_nodes))
        # Inicialization of weights
        for i in range(len(self.weights_input_hidden)):
            self.weights_input_hidden[i] = np.random.uniform(-1,1)
            for j in range(len(self.weights_input_hidden)):
                self.weights_input_hidden[i][j] = np.random.uniform(-1,1)
        for i in range(len(self.weights_hidden_output)):
            self.weights_hidden_output[i] = np.random.uniform(-1,1)
            for j in range(len(self.weights_hidden_output)):
                self.weights_hidden_output[i][j] = np.random.uniform(-1,1)
                
        self.bias_hidden = np.empty((self.hidden_nodes,1))
        self.bias_output = np.empty((self.output_nodes,1))
        # Inicialization of biases
        for i in range(len(self.bias_hidden)):
            self.bias_hidden[i] = np.random.uniform(0,1)
        for i in range(len(self.bias_output)):
            self.bias_output[i] = np.random.uniform(0,1)    
        
    def feedforward(self, input):

        # Generating hidden outputs
        hidden = np.dot(self.weights_input_hidden, input) + self.bias_hidden 
        # Activation function
        hidden = np.multiply(hidden, sigmoid(hidden[0][0]))

        # Generating outputs
        output = np.dot(self.weights_hidden_output, hidden) + self.bias_output
        # Activation function
        output = np.multiply(output,sigmoid(output[0][0]))
        output = np.asarray(output).ravel()
        return output
    
    def train(self, inputs, answer):
        pass
        