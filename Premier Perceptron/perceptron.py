import random


def output(somme): #Activation function

    if somme >= 0:
        return 1
    else:
        return -1


class Perceptron:


    def __init__(self,nb_variable):

        self.weights = []
        i = 0
        while i < nb_variable:
            self.weights.append(random.uniform(-1,1))
            i += 1

    def guess(self,point): # (x,y)

        somme = 0
        i = 0

        for weight in self.weights:
            somme += point.position[i] * weight
            i += 1

        return output(somme)

    def train(self,point):
        guessed = self.guess(point)
        error =  point.reponse - guessed
        lr = 1
        i = 0

        for weight in self.weights:
            self.weights[i] += error * point.position[i] * lr

            i += 1












