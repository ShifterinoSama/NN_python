import nn


# OPRAVIT NEFUNGUJE JDU SPAT
td = [
    [[0,1],1],
    [[1,0],1],
    [[0,0],0],
    [[1,1],0]
]

brain = nn.Neural_Network(2,2,1)

for i in range(101):
    brain.train(td[0][0],td[0][1])
    brain.train(td[1][0],td[1][1])
    brain.train(td[2][0],td[2][1])
    brain.train(td[3][0],td[3][1])

print(brain.feedforward([1,0]))
print(brain.feedforward([0,1]))
print(brain.feedforward([0,0]))
print(brain.feedforward([1,1]))

#inputs = [1,0]
#targets = [1,0]
#brain.train(inputs,targets)
