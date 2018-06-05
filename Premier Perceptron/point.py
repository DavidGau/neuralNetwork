import random

class Point:

    def __init__(self,width,height):
        self.position  = (random.randint(0,width),random.randint(0,height))

        if self.position[0] > self.position[1]:
            self.reponse = 1
        else:
            self.reponse = -1
