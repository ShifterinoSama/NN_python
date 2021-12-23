import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neural_Network():
    def __init__(self,input_nodes, hidden_nodes, output_nodes) -> None:
        self.input_nodes = input_nodes 
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Inicialization of weights
        self.weights_input_hidden = np.random.uniform(-1,1,size=(hidden_nodes, input_nodes))
        self.weights_hidden_output = np.random.uniform(-1,1,size=(output_nodes, hidden_nodes))

        # Inicialization of biases    
        self.bias_hidden = np.random.uniform(0,1,size=(self.hidden_nodes,1))
        self.bias_output = np.random.uniform(0,1,size=(self.output_nodes,1))
           
        
    def feedforward(self, inputs):
        # Generating hidden outputs
        hidden = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden 
        # Activation function
        hidden = np.multiply(hidden, sigmoid(hidden[0][0]))

        # Generating outputs
        output = np.dot(self.weights_hidden_output, hidden) + self.bias_output
        # Activation function
        output = np.multiply(output,sigmoid(output[0][0]))
        output = np.asmatrix(output)
        return output.sum(axis=1)
    
    def train(self, inputs, targets):
        outputs = self.feedforward(inputs)
        # Calculate the error
        # ERROR = TARGETS - OUTPUTS
        output_errors = np.subtract(targets, outputs)

        # Zkontrolovat jestli sedí dimenze a není nutný transpose
        hidden_errors = np.dot(self.weights_hidden_output, output_errors)
          
        pass
        