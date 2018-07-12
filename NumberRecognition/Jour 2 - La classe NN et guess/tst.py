import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_p(x):
  return x*(1-x)
i = 3

i = sigmoid(i)
print(i)

print(sigmoid_p(i))
