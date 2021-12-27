import nn
import random

training_data = [[[0,1],[1]],
                [[0,0],[0]],
                [[1,0],[1]],
                [[1,1],[0]]]

t10,t01,f00,f11 = 0, 0, 0, 0

brain = nn.Neural_Network(2,4,1)

for i in range(10001):
    data = random.choice(training_data)
    if data[0] == [0,1]:
        t01 += 1
    elif data[0] == [1,0]:
        t10 += 1
    elif data[0] == [0,0]:
        f00 += 1
    elif data[0] == [1,1]:
        f11 += 1
    brain.train(data[0],data[1])

print("Number of - 1,0 - in training: ",t10)
print("Number of - 0,1 - in training: ",t01)
print("Number of - 0,0 - in training: ",f00)
print("Number of - 1,1 - in training: ",f11)
print(brain.feedforward([1,0]), "Should be True")
print(brain.feedforward([0,1]), "Should be True")
print(brain.feedforward([0,0]), "Should be False")
print(brain.feedforward([1,1]), "Should be False")


