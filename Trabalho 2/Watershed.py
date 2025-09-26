# Gustavo Guerreiro e João Martinho

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Função principal para segmentar núcleos usando watershed
def segmentar_nucleos(imagem_path):
    # 1. Carregar a imagem
    img = cv2.imread(imagem_path)
    if img is None:
        raise ValueError("Imagem não encontrada ou caminho inválido")
    
    # 2. Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Aplicar filtro Gaussiano para remover ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Aplicar limiar adaptativo para segmentação inicial
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Remover pequenos ruídos usando morfologia (abertura)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 6. Determinar área segura (foreground) usando dilatação
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 7. Calcular área segura para os núcleos usando distância transform e threshold
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

    # 8. Encontrar região desconhecida (fronteira entre foreground e background)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 9. Marcar os marcadores para watershed
    ret, markers = cv2.connectedComponents(sure_fg)

    # 10. Incrementar marcadores para que o background seja 1 em vez de 0
    markers = markers + 1

    # 11. Marcar região desconhecida com zero
    markers[unknown == 255] = 0

    # 12. Aplicar watershed
    markers = cv2.watershed(img, markers)

    # 13. Marcar bordas encontradas pelo watershed na imagem original
    img[markers == -1] = [0, 0, 255]  # vermelho nas bordas

    # Exibir resultado
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Segmentação com Watershed'), plt.axis('off')
    plt.subplot(122), plt.imshow(markers, cmap='jet')
    plt.title('Marcadores Watershed'), plt.axis('off')
    plt.show()

    return markers

# Exemplo de uso
# Altere 'caminho/para/imagem.jpg' para o caminho da sua imagem do dataset
segmentar_nucleos('dataset_neutrofilos/neutrofilo06.png')
