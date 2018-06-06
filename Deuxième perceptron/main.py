import Perceptron
import Point
import Canvas
import time

canvas_width = 600
canvas_height = 600
points_actif = []
points_avant = 1 #Lors du clear du canvas, les new elements continuent selon les anciens id
perceptron = Perceptron.Perceptron(2)


#Detecte le clique gauche et fait action en conséquence
def clique_gauche(d):
    cree_point(50)

#Detecte le clique sur la molette et fait action en conséquence
def clique_molette(d):
    supprime_point()

#Detecte le clique droit et fait action en conséquence
def clique_droit(d):
    guess_point()

#Detecte le space bar et fait action en conséquence
def appuye_space(d):
    entraine_point()

canvas = Canvas.Canvas(canvas_width,canvas_height,clique_gauche,clique_molette,clique_droit,appuye_space)



#Cree le nombre de pt demandé et les affiches
def cree_point(nb):
    for i in range(0,nb):
        point_cree = Point.Point(canvas_width,canvas_height)
        canvas.cree_cercle(point_cree.coord,25,"grey",point_cree.label)
        points_actif.append(point_cree)
    canvas.cree_ligne(canvas_width,canvas_height,5)



#Le Perceptron fait un guess sur la position des points
def guess_point():
    nb_erreur = 0
    for i in range(0,len(points_actif)):
        guess = perceptron.guess(points_actif[i])

        if guess != points_actif[i].label:
            canvas.modifie_couleur_cercle(i + points_avant,"red")
            nb_erreur += 1
        else:
            canvas.modifie_couleur_cercle(i + points_avant,"green")
    print("Erreur lors du guess: ",nb_erreur)

#Entraîne le Perceptron sur tout les pt actifs
def entraine_point():
    for i in range(0,len(points_actif)):
        perceptron.train(points_actif[i])
    guess_point()

#Supprime les points
def supprime_point():
    global points_avant
    global points_actif

    canvas.clear_canvas()
    points_avant += len(points_actif) + 1
    points_actif = []




