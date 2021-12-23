import nn

brain = nn.Neural_Network(2,2,2)

inputs = [1,0]
targets = [1,0]
output = brain.feedforward(inputs)
brain.train(inputs,targets)
