from mnist import MNIST
from matplotlib import pyplot as plt
import numpy as np


#Charge les images
mndata = MNIST('')
images, labels = mndata.load_training()


#Affiche l'opacité des pixels de l'image
images[10] = [round(color / 255,2) for color in images[0]]
print(images[10])




#Display l'image
first_image = images[10]
first_image = np.array(first_image, dtype='float')
pixels = first_image.reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

