# Gustavo Guerreiro e João Martinho

import numpy as np
from sklearn.cluster import MeanShift
import cv2
from matplotlib import pyplot as plt

def to_data(img):
    if img.ndim == 2:  # imagem em tons de cinza
        return img.reshape(-1, 1).astype(float)
    return img.reshape(-1, img.shape[2]).astype(float)

def mean_shift(data, bandwidth=15):
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    return labels, cluster_centers

def make_shifted_img(shape, labels, centers):
    labels = np.array(labels)
    img = centers[labels].reshape(shape).astype(np.uint8)
    return img

def process_image(path, bandwidth=17, crop_fraction=(0.0, 1.0, 0.0, 1.0)):
    """
    crop_fraction: tupla (top_frac, bottom_frac, left_frac, right_frac)
    Valores de 0.0 a 1.0 para indicar a fração da imagem a ser recortada
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    # Se imagem tiver canal alfa, descarta
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Calcula crop baseado em frações
    h, w = img.shape[:2]
    top = int(h * crop_fraction[0])
    bottom = int(h * crop_fraction[1])
    left = int(w * crop_fraction[2])
    right = int(w * crop_fraction[3])
    img = img[top:bottom, left:right]

    if img.size == 0:
        raise ValueError("O recorte resultou em uma imagem vazia")

    # Converte BGR -> RGB se necessário
    if img.ndim == 3 and img.shape[2] == 3:
        img = img[:, :, ::-1]

    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    plt.title("Imagem original recortada")
    plt.show()

    img_data = to_data(img)
    print("Dados da imagem formatados")

    img_labels, img_centers = mean_shift(img_data, bandwidth=bandwidth)
    print("Labels e centros obtidos")

    new_img = make_shifted_img(img.shape, img_labels, img_centers)
    print("Nova imagem gerada")

    plt.imshow(new_img, cmap='gray' if new_img.ndim == 2 else None)
    plt.title("Imagem após MeanShift")
    plt.show()

    return new_img

# Exemplo de uso: pega toda a imagem
new_img = process_image('./dataset_neutrofilos/neutrofilo06.png', bandwidth=17)
