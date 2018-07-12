import numpy
from mnist import MNIST
from matplotlib import pyplot as plt
import numpy as np
import time
"""
    Classe servant à crée des Neural Networks
    Elle sera au début hardcoder en grande partie
    mais deviendra de plus en plus modulables
"""

class NeuralNetwork:

    #Constructeur
    def __init__(self):

        nb_node_input = 784

        nb_node_hl1 = 16
        nb_node_hl2 = 16

        nb_node_output = 10

        #Matrices input-hl1
        self.weights_input_hl1 = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_node_hl1,nb_node_input)))
        self.bias_input_hl1 = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_node_hl1,1)))

        #Matrices hl1-hl2
        self.weights_hl1_hl2 = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_node_hl2,nb_node_hl1)))
        self.bias_hl1_hl2 = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_node_hl2,1)))

        #Matrices hl2-output
        self.weights_hl2_output = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_node_output,nb_node_hl2)))
        self.bias_hl2_output = numpy.asmatrix(numpy.random.uniform(-1,1,(nb_node_output,1)))


    #Fait un guess
    def guess(self,features):

        features = numpy.asmatrix(features) #Vecteur des features

        #Weighted sum de input à hl1
        sum_input_hl1 = numpy.dot(self.weights_input_hl1,features)
        sum_input_hl1 = numpy.add(sum_input_hl1,self.bias_input_hl1)
        sum_input_hl1 = self.sigmoid(sum_input_hl1)
        #Weighted sum de hl1 à hl2
        sum_input_hl2 = numpy.dot(self.weights_hl1_hl2,sum_input_hl1)
        sum_input_hl2 = numpy.add(sum_input_hl2,self.bias_hl1_hl2)
        sum_input_hl2 = self.sigmoid(sum_input_hl2)
        #Weighted sum de h2 à output
        sum_output = numpy.dot(self.weights_hl2_output,sum_input_hl2)
        sum_output = numpy.add(sum_output,self.bias_hl2_output)


        return self.sigmoid(sum_output)

    #Essaye d'entraîner le network
    def train(self,features,label):
        features = numpy.asmatrix(features) #Vecteur des features
        labels = numpy.asmatrix(label)

        #Weighted sum de input à hl1
        sum_input_hl1 = numpy.dot(self.weights_input_hl1,features)
        sum_input_hl1 = numpy.add(sum_input_hl1,self.bias_input_hl1)
        sum_input_hl1 = self.sigmoid(sum_input_hl1)
        #Weighted sum de hl1 à hl2
        sum_input_hl2 = numpy.dot(self.weights_hl1_hl2,sum_input_hl1)
        sum_input_hl2 = numpy.add(sum_input_hl2,self.bias_hl1_hl2)
        sum_input_hl2 = self.sigmoid(sum_input_hl2)
        #Weighted sum de h2 à output
        sum_output = numpy.dot(self.weights_hl2_output,sum_input_hl2)
        sum_output = numpy.add(sum_output,self.bias_hl2_output)
        sum_output = self.sigmoid(sum_output)
        ######################################
        #              Backprop              #
        ######################################

        label_matrix = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
        label_matrix[label] = [1]

        erreurs_output = numpy.subtract(label_matrix,sum_output)
        erreurs_output = self.sigmoid_prime(erreurs_output)
        #Commencement de la back propagation
        #Calcul du delta weight Output
        somme_weights_output_row = numpy.sum(self.weights_hl2_output,axis=1)

        delta_weight_output = numpy.divide(self.weights_hl2_output,somme_weights_output_row)
        delta_weight_output = numpy.multiply(delta_weight_output,erreurs_output)

        #Calcul des erreurs par nodes avant donc, le cycle pourra reprendre
        erreurs_hl2 = numpy.transpose(delta_weight_output)
        erreurs_hl2 = numpy.sum(erreurs_hl2,axis=1)

        #Ajout de la learning rate
        delta_weight_output = numpy.multiply(delta_weight_output,0.0001)
        self.weights_hl2_output = numpy.add(self.weights_hl2_output,delta_weight_output)

        delta_bias_output = numpy.multiply(erreurs_output,0.0001)
        self.bias_hl2_output = numpy.add(self.bias_hl2_output,delta_bias_output)

        #CALCUL DU DELTA WEIGHT HL2-HL1
        erreurs_hl2 = self.sigmoid_prime(erreurs_hl2)

        somme_weights_h2_row = numpy.sum(self.weights_hl1_hl2,axis=1)

        delta_weight_h2 = numpy.divide(self.weights_hl1_hl2,somme_weights_h2_row)
        delta_weight_h2 = numpy.multiply(delta_weight_h2,erreurs_hl2)

        #Erreurs par nodes HL1
        erreurs_hl1 = numpy.transpose(delta_weight_h2)
        erreurs_hl1 = numpy.sum(erreurs_hl1,axis=1)

        #Ajout de la learning rate
        delta_weight_h2 = numpy.multiply(delta_weight_h2,0.0001)
        self.weights_hl1_hl2 = numpy.add(self.weights_hl1_hl2,delta_weight_h2)

        delta_bias_hl2 = numpy.multiply(erreurs_hl2,0.0001)
        self.bias_hl1_hl2 = numpy.add(self.bias_hl1_hl2,delta_bias_hl2)
        #CALCUL DU DELTA WEIGHT HL1-INPUT
        erreurs_hl1 = self.sigmoid_prime(erreurs_hl1)

        somme_weights_h1_row = numpy.sum(self.weights_input_hl1,axis=1)

        delta_weights_hl1 = numpy.divide(self.weights_input_hl1,somme_weights_h1_row)
        delta_weights_hl1 = numpy.multiply(delta_weights_hl1,erreurs_hl1)

        delta_weights_hl1 = numpy.multiply(delta_weights_hl1,0.0001)

        self.weights_input_hl1 = numpy.add(self.weights_input_hl1,delta_weights_hl1)

        delta_bias_hl1 = numpy.multiply(erreurs_hl2,0.0001)
        self.bias_input_hl1 = numpy.add(self.bias_input_hl1,delta_bias_hl1)
        #DONE



    def sigmoid(self,x):
      return 1 / (1 + numpy.exp(-x))

    def sigmoid_prime(self,y):
        m_y = numpy.subtract(y,1)
        return numpy.multiply(y,m_y)

#Charge les images
mndata = MNIST('')
images, labels = mndata.load_training()




first_run = NeuralNetwork()


for i in range(0,10000):
    time.sleep(0.00000001)
    image_envoye = [round(color / 255,2) for color in images[i]]
    image_envoye = [[nombre] for nombre in images[i]]
    first_run.train(image_envoye,labels[i])
print("Fini! il est supposément trained")


bonne_reponse = 0
mauvaise_reponse = 0

for j in range(1000,1500):
    image_envoye = [round(color / 255,2) for color in images[j]]
    image_envoye =  [[nombre] for nombre in images[j]]
    guess = first_run.guess(image_envoye)
    guess = numpy.asarray(guess)
    plushaut = 0
    index = 0
    ij = 0
    for nb in guess:
        if nb > plushaut:
            plushaut = nb
            index = ij
        ij += 1

    if index == labels[j]:
        bonne_reponse += 1
    else:
        mauvaise_reponse += 1


print("Le NN a eu: ",bonne_reponse)
print("Le NN a eu: ",mauvaise_reponse)
print("Moyenne: ",bonne_reponse / mauvaise_reponse)


#Display l'image
first_image = images[500]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()
