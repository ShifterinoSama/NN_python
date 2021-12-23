import nn

brain = nn.Neural_Network(2,2,1)
input = [1,0]
output = brain.feedforward(input)
print(output)
