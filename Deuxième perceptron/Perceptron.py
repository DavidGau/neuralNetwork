import random

class Perceptron:
    """
        L'objet perceptron peut être vue comme
        étant une neuronne
    """

    #Constructeur
    def __init__(self,nb_weights):
        #inititalise les weights au hasard
        self.weights = []
        self.learning_rate = 0.1

        for i in range(0,nb_weights):
            self.weights.append(random.uniform(-1,1))

    #Permet de faire un guess sans training
    def guess(self,point):
        somme = 0

        for i in range(0,len(self.weights)):
            somme += point.coord[i] * self.weights[i]

        guess_formater = self.format_output(somme)
        return guess_formater

    #Permet de formater la reponse du guess
    def format_output(self,n):
        if n >= 0:
            return 1
        else:
            return -1

    #Permet de train le Perceptron, change donc les weights selon l'erreur
    def train(self,point):
        guess = self.guess(point)
        reponse = point.label

        erreur = reponse - guess

        for i in range(0,len(self.weights)): #Utilie se type de for car sinon l'objet retourné par la for est invariable
            self.weights[i] += erreur * point.coord[i] * self.learning_rate

        #print("w1: {} w2: {}".format(self.weights[0],self.weights[1]))











