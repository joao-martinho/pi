# Gustavo Guerreiro e João Martinho

import matplotlib.pyplot as plt
import cv2
import numpy as np

img = cv2.imread('./dataset_neutrofilos/neutrofilo06.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converte para RGB para matplotlib

Z = img.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 7
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

plt.imshow(res2)
plt.axis('off')  # opcional: remove eixos
plt.show()
