import numpy as np

def vectorized_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def vectorized_d_sigmoid(y):
    return y * (1-y)

class Neural_Network():
    def __init__(self,input_nodes, hidden_nodes, output_nodes, learning_rate=0.1) -> None:
        self.input_nodes = input_nodes 
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Inicialization of weights
        self.weights_input_hidden = np.array(np.random.uniform(-1,1,size=(self.hidden_nodes, self.input_nodes)))
        self.weights_hidden_output = np.array(np.random.uniform(-1,1,size=(self.output_nodes, self.hidden_nodes)))
        # Inicialization of weights
        #self.weights_input_hidden = np.random.uniform(-1,1,size=(hidden_nodes, input_nodes))
        #self.weights_hidden_output = np.random.uniform(-1,1,size=(output_nodes, hidden_nodes))

        # Inicialization of biases 
        self.bias_hidden = np.array(np.random.uniform(0,1,size=(self.hidden_nodes,1)))
        self.bias_output = np.array(np.random.uniform(0,1,size=(self.output_nodes,1)))

        # Inicialization of biases    
        #self.bias_hidden = np.random.uniform(0,1,size=(self.hidden_nodes,1))
        #self.bias_output = np.random.uniform(0,1,size=(self.output_nodes,1))
       
    def from_array(self, arr):
        m = np.ndarray((len(arr), 1))
        for i in range(len(arr)):
            m[i][0] = arr[i]
        return m

    def to_array(self, arr):
        new_arr = []
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                new_arr.append(arr[i][j])
        return new_arr

    def feedforward(self, inputs):

        inputs = self.from_array(inputs)
        # Generating hidden outputs
        hidden = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden
        # Activation function
        hidden = vectorized_sigmoid(hidden)
        # Generating outputs
        outputs = np.dot(self.weights_hidden_output,hidden) + self.bias_output
        # Activation function
        outputs = vectorized_sigmoid(outputs)

        return self.to_array(outputs)
    
    def train(self, inputs, targets):

        inputs = self.from_array(inputs)

        # Generating hidden outputs
        hidden = np.dot(self.weights_input_hidden, inputs) + self.bias_hidden 
        # Activation function
        hidden = vectorized_sigmoid(hidden)
        # Generating outputs
        outputs = np.dot(self.weights_hidden_output, hidden) + self.bias_output
        # Activation function
        outputs = vectorized_sigmoid(outputs)
        # Convert array to np.matrix object
        targets = self.from_array(targets)
        # Calculate the error | Erorr = targets - outputs
        output_errors = np.subtract(targets, outputs)
        # Calculate gradient
        # Gradient = outputs * (1-outuputs)
        gradients = vectorized_d_sigmoid(outputs)
        gradients = np.multiply(gradients, output_errors)
        gradients = np.multiply(gradients, self.learning_rate)
        # Calculate deltas
        hidden = np.asarray(hidden)
        hidden_T = np.transpose(hidden)
        weights_hidden_output_deltas = np.multiply(gradients, hidden_T)
        # Adjust weights by deltas
        self.weights_hidden_output = np.add(self.weights_hidden_output, weights_hidden_output_deltas)
        # Adjust the bias by its deltas
        self.bias_output = np.add(self.bias_output, gradients)
        # Calculate the hidden layer errors
        weights_hidden_output_T = np.transpose(self.weights_hidden_output)
        hidden_errors = np.multiply(weights_hidden_output_T, output_errors)
        # Calculate hidden gradient
        hidden_gradients = vectorized_d_sigmoid(hidden)
        hidden_gradients = np.multiply(hidden_gradients, hidden_errors)
        hidden_gradients = np.multiply(hidden_gradients, self.learning_rate)
        # Calculate input->hidden deltas
        inputs_T = np.transpose(inputs)
        weights_input_hidden_deltas = np.multiply(hidden_gradients, inputs_T)
        # Adjust weights by deltas
        self.weights_input_hidden = np.add(self.weights_input_hidden, weights_input_hidden_deltas)
        # Adjust the bias by its deltas
        self.bias_hidden = np.add(self.bias_hidden, hidden_gradients)
 