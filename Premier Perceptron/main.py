import perceptron
from point import *
from tkinter import *



canvas_width = 700
canvas_height = 700

brain = perceptron.Perceptron(2)
points = []
point_graphique = []
nb_point = 1000

i = 0

while i < nb_point:
    new_point = Point(canvas_width,canvas_height)
    points.append(new_point)
    i += 1

fenetre = Tk()
canvas = Canvas(fenetre,width = canvas_width,height = canvas_height)

for p in points:
    if p.reponse == 1:
        objet = canvas.create_oval(p.position[0],p.position[1],p.position[0] + 25,p.position[1] + 25,width=2)
    else:
        objet = canvas.create_oval(p.position[0],p.position[1],p.position[0] + 25,p.position[1] + 25,width=10)



    point_graphique.append(objet)

canvas.create_line(0,0,canvas_width,canvas_height,width=2)

y = 0
for g in points:

    brain.guess(g)
    if brain.guess(g) == g.reponse:
        canvas.itemconfigure(point_graphique[y],fill="green")
    else:
        canvas.itemconfigure(point_graphique[y],fill="red")
    y += 1

def tst(fd):
    q = 0
    print("training")


    for p in points:

       brain.train(p)
       q = 0

       error = 0
       for h in points:
           if brain.guess(h) == h.reponse:
               canvas.itemconfigure(point_graphique[q],fill="green")
           else:
               canvas.itemconfigure(point_graphique[q],fill="red")
               error += 1
           q += 1


       canvas.pack()
    print("Erreur restante:",error)



    """ Version 1

    for p in points:

        brain.train(p)

        if brain.guess(p) == p.reponse:
            canvas.itemconfigure(point_graphique[q],fill="green")
        else:
            canvas.itemconfigure(point_graphique[q],fill="red")
            error += 1
        q += 1
    canvas.pack()


    """


fenetre.bind("<Button-1>", tst)
canvas.pack()
fenetre.mainloop()




