import nn
import random

training_data = [[[0,1],[1]],
                [[1,0],[1]],
                [[0,0],[0]],
                [[1,1],[0]]]

brain = nn.Neural_Network(2,2,1,0.1)

for i in range(100001):
    data = random.choice(training_data)
    brain.train(data[0],data[1])

print(brain.feedforward([1,0]))
print(brain.feedforward([0,1]))
print(brain.feedforward([0,0]))
print(brain.feedforward([1,1]))


