# %pip install matplotlib
# %pip install opencv-python
# %pip install numpy

from matplotlib import pyplot as plt, cm
import cv2
import numpy as np

imagem = cv2.imread('Santarem3semagua.tif')
if len(imagem.shape) == 3:
    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

plt.axis('off')
plt.title('Santarém Banda 3 Sem Água')
plt.imshow(imagem, 'gray')
# plt.show()

(T, binaria) = cv2.threshold(imagem, 33, 255, cv2.THRESH_BINARY)
plt.figure(figsize=(10, 5))
plt.title('santarem_binarizada')
plt.axis("off")
plt.imshow(binaria, 'gray')

kernel = np.array([
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0]
], dtype=np.uint8)

erodida = cv2.erode(binaria, kernel, iterations=2)

plt.axis('off')
plt.title("Após erosão")
plt.imshow(erodida, 'gray')
# plt.show()

bordas, hierarquia = cv2.findContours(erodida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

imagem_colorida = cv2.cvtColor(erodida, cv2.COLOR_GRAY2BGR)
cv2.drawContours(imagem_colorida, bordas, -1, (0,0,255), thickness=2)

plt.title("Imagem erodida + contornos vermelhos")
plt.axis("off")
plt.imshow(cv2.cvtColor(imagem_colorida, cv2.COLOR_BGR2RGB))
# plt.show()

imagem = cv2.imread("Santarem345.tif", cv2.IMREAD_UNCHANGED).astype(np.float32)
imagem_norm = (imagem - imagem.min()) / (imagem.max() - imagem.min())
pseudo = (cm.get_cmap("terrain")(imagem_norm)[:, :, :3] * 255).astype(np.uint8)

alpha = 0.4
resultado = cv2.addWeighted(
    cv2.drawContours(pseudo.copy(), bordas, -1, (255, 0, 0), thickness=cv2.FILLED),
    alpha,
    pseudo,
    1 - alpha,
    0
)

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Santarem345 com máscaras vermelhas sobre o verde")
plt.imshow(resultado)
plt.show()
