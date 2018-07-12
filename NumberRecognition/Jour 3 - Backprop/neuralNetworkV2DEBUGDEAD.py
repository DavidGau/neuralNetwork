import numpy
from mnist import MNIST
from matplotlib import pyplot as plt
import numpy as np
import time
import random
"""
    Classe servant à crée des Neural Networks
    Elle sera au début hardcoder en grande partie
    mais deviendra de plus en plus modulables
"""

class NeuralNetwork:

    #Constructeur
    def __init__(self):

        nb_node_input = 2

        nb_node_hl1 = 2

        nb_node_output = 1

        #Matrices input-hl1
        self.weights_input_hl1 = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_node_hl1,nb_node_input)))
        self.bias_input_hl1 = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_node_hl1,1)))

        #Matrices hl2-output
        self.weights_hl1_output = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_node_output,nb_node_hl1)))
        self.bias_hl1_output = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_node_output,1)))


    #Fait un guess
    def guess(self,features):

        features = numpy.asmatrix(features) #Vecteur des features

        #Weighted sum de input à hl1
        sum_input_hl1 = numpy.dot(self.weights_input_hl1,features)
        sum_input_hl1 = numpy.add(sum_input_hl1,self.bias_input_hl1)

        #Weighted sum de h2 à output
        sum_output = numpy.dot(self.weights_hl1_output,sum_input_hl1)
        sum_output = numpy.add(sum_output,self.bias_hl1_output)


        return self.sigmoid(sum_output)

    #Essaye d'entraîner le network
    def train(self,features,label):

        features = numpy.asmatrix(features) #Vecteur des features

        #Weighted sum de input à hl1
        sum_input_hl1 = numpy.dot(self.weights_input_hl1,features)
        sum_input_hl1 = numpy.add(sum_input_hl1,self.bias_input_hl1)

        #Weighted sum de h2 à output
        sum_output = numpy.dot(self.weights_hl1_output,sum_input_hl1)
        sum_output = numpy.add(sum_output,self.bias_hl1_output)
        sum_output = self.sigmoid(sum_output)
        ######################################
        #              Backprop              #
        ######################################

        erreurs_output = label - sum_output
        erreurs_output = self.sigmoid_prime(erreurs_output)

        #Commencement de la back propagation
        #Calcul du delta weight Output
        somme_weights_output_row = numpy.sum(self.weights_hl1_output,axis=1)

        delta_weight_output = numpy.divide(self.weights_hl1_output,somme_weights_output_row)
        delta_weight_output = numpy.multiply(delta_weight_output,erreurs_output)

        #Calcul des erreurs par nodes avant donc, le cycle pourra reprendre
        erreurs_hl1 = numpy.transpose(delta_weight_output)
        erreurs_hl1 = numpy.sum(erreurs_hl1,axis=1)

        #Ajout de la learning rate
        delta_weight_output = numpy.multiply(delta_weight_output,0.00001)

        self.weights_hl1_output = numpy.add(self.weights_hl1_output,delta_weight_output)

        delta_bias_output = numpy.multiply(erreurs_output,0.00001)

        self.bias_hl1_output = numpy.add(self.bias_hl1_output,delta_bias_output)


        somme_weights_h1_row = numpy.sum(self.weights_input_hl1,axis=1)

        delta_weights_hl1 = numpy.divide(self.weights_input_hl1,somme_weights_h1_row)
        delta_weights_hl1 = numpy.multiply(delta_weights_hl1,erreurs_hl1)

        delta_weights_hl1 = numpy.multiply(delta_weights_hl1,0.00001)

        self.weights_input_hl1 = numpy.add(self.weights_input_hl1,delta_weights_hl1)

        delta_bias_hl1 = numpy.multiply(erreurs_hl1,0.00001)

        self.bias_input_hl1 = numpy.add(self.bias_input_hl1,delta_bias_hl1)

        #DONE



    def sigmoid(self,x):
      return 1 / (1 + numpy.exp(-x))

    def sigmoid_prime(self,y):
        m_y = numpy.subtract(y,1)
        return numpy.multiply(y,m_y)



first_time = NeuralNetwork()
first_time.train([[1],[0]],[1])



data_set = [
    [[[1],[0]],[1]],
    [[[0],[1]],[1]],
    [[[0],[0]],[0]],
    [[[1],[1]],[0]],
]
print(data_set[0][0])
print(data_set[0][1][0])
print(first_time.train(data_set[0][0],data_set[0][1][0]))

for i in range(0,18000):
    time.sleep(0.0000001)
    random.shuffle(data_set)
    first_time.train(data_set[0][0],data_set[0][1][0])
    first_time.train(data_set[1][0],data_set[1][1][0])
    first_time.train(data_set[2][0],data_set[2][1][0])
    first_time.train(data_set[3][0],data_set[3][1][0])
print("=====================================================")

