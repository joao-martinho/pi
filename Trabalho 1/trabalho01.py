from matplotlib import pyplot as plt
import cv2
import numpy as np

# carrega imagem, aplica binzarização
imagem = cv2.imread('santarem.png')
imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
(T, binarizacao) = cv2.threshold(imagem, 140, 255, cv2.THRESH_BINARY)
plt.subplot(1, 3, 1)
plt.imshow(binarizacao, 'gray')
plt.title('santarem_binarizada')

# erosão
kernel = np.ones((5, 5), np.uint8)
erosao = cv2.erode(binarizacao, kernel, iterations = 1)
plt.imshow(erosao, 'gray')
plt.title('santarem_erodida')

# segmentação
segmentacao = erosao.copy()
_, mask = cv2.threshold(erosao, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# dilatação
kernel = np.ones((5, 5), np.uint8)
dilatacao = cv2.dilate(imagem, kernel, iterations = 1)
plt.imshow(dilatacao, 'gray')
plt.title('santarem_dilatada')

plt.show()
