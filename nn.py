import numpy as np
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Neural_Network():
    def __init__(self,input_nodes, hidden_nodes, output_nodes, learning_rate=0.01) -> None:
        self.input_nodes = input_nodes 
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

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
        outputs = np.dot(self.weights_hidden_output, hidden) + self.bias_output
        # Activation function
        outputs = np.multiply(outputs,sigmoid(outputs[0][0]))
        outputs_final = np.asmatrix(outputs)
        outputs_final = outputs_final.sum(axis=1)
        return outputs_final
    
    def train(self, inputs, targets):
        # Generating hidden outputs
        hidden = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden 
        # Activation function
        hidden = np.multiply(hidden, sigmoid(hidden[0][0]))

        # Generating outputs
        outputs = np.dot(self.weights_hidden_output, hidden) + self.bias_output
        # Activation function
        outputs = np.multiply(outputs,sigmoid(outputs[0][0]))
        outputs = np.asmatrix(outputs)
        outputs = outputs.sum(axis=1)

        # Calculate the error
        # ERROR = TARGETS - OUTPUTS
        output_errors = np.subtract(targets, outputs)

        # Calculate gradient
        gradients = np.multiply(outputs,(1-outputs))
        gradients = np.multiply(gradients, output_errors)
        gradients = np.multiply(gradients,self.learning_rate)

        # Calculate deltas
        weight_hidden_output_delta = np.multiply(hidden,gradients)
        # Adjust the weights by deltas
        self.weights_hidden_output = np.add(weight_hidden_output_delta,self.weights_hidden_output)
        # Adjust the bias by its deltas
        self.bias_output = np.add(gradients,self.bias_output)

        # Zkontrolovat jestli sedí dimenze a není nutný transpose
        hidden_errors = np.multiply(self.weights_hidden_output, output_errors)
        
        # Calculate hidden gradient
        hidden_gradients = np.multiply(hidden,(1-hidden))
        hidden_gradients = np.multiply(hidden_gradients,hidden_errors)
        hidden_gradients = np.multiply(hidden_gradients,self.learning_rate)

        # calculate input->hidden deltas
        weight_input_hidden_delta = np.dot(hidden_gradients,inputs)
        # Adjust the hidden_weights by hidden_deltas
        self.weights_input_hidden = np.add(weight_input_hidden_delta, self.weights_input_hidden)
        # Adjust the bias by its deltas
        self.bias_hidden = np.add(hidden_gradients,self.bias_hidden)

        