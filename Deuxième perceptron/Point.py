import random

class Point:
    """
        Sert à définir un point pour et sans le training
        coord,label
    """

    #Constructeur
    def __init__(self,canvas_width,canvas_height):
        #Assigne des coordonnées random selon la width et l'height données
        self.coord = (random.uniform(0,canvas_width),random.uniform(0,canvas_height))

        if self.coord[0] > self.coord[1]: #Selon la règle de l'équation
            self.label = 1
        else:
            self.label = -1
