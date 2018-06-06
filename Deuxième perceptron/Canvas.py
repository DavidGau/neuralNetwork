import tkinter

class Canvas:
    """
        Sert à crée l'interface graphique du Perceptron
        peut crée des ligne, cercles graces aux méthodes de Canvas,
        peut également modifier les propriétés des objets crées
    """


    #Constructeur, initialise la taille de la fenetre et crée le canvas
    def __init__(self,canvas_width,canvas_height,fct1,fct2,fct3,fct4):
        self.fenetre = tkinter.Tk()
        self.canvas = tkinter.Canvas(self.fenetre,width = canvas_width,height = canvas_height)
        self.canvas.pack()
        self.fenetre.bind("<Button-1>", fct1)
        self.fenetre.bind("<Button-2>", fct2)
        self.fenetre.bind("<Button-3>", fct3)
        self.fenetre.bind("<space>", fct4)
    #Sert à crée un cercle
    def cree_cercle(self,coord,circle_width,color,border_width):
        border_width = 10 if border_width >= 0 else 4
        self.canvas.create_oval(coord[0],coord[1],coord[0] + circle_width,coord[1] + circle_width,fill=color,width=border_width)
        self.canvas.pack()

    #Sert à modifier la couleur du cercle
    def modifie_couleur_cercle(self,id,color):
        self.canvas.itemconfigure(id,fill=color)
        self.canvas.pack()

    #Sert à crée une ligne
    def cree_ligne(self,canvas_width,canvas_height,line_width):
        self.canvas.create_line(canvas_width,canvas_height,0,0,width=line_width)
        self.canvas.pack()

    #Sert à supprimer tout ce qui se trouve sur le canvas
    def clear_canvas(self):
        self.canvas.delete("all")

