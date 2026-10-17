import cv2
import numpy as np

def update_mask(val):
    # Obter os valores dos sliders
    r_min = cv2.getTrackbarPos("R Min", "Sliders")
    r_max = cv2.getTrackbarPos("R Max", "Sliders")
    g_min = cv2.getTrackbarPos("G Min", "Sliders")
    g_max = cv2.getTrackbarPos("G Max", "Sliders")
    b_min = cv2.getTrackbarPos("B Min", "Sliders")
    b_max = cv2.getTrackbarPos("B Max", "Sliders")

    # Criar os limites para a máscara
    lower_bound = np.array([r_min, g_min, b_min])
    upper_bound = np.array([r_max, g_max, b_max])

    # Criar a máscara para os pixels dentro do intervalo RGB
    mask = cv2.inRange(image_rgb, lower_bound, upper_bound)

    # Aplicar a máscara na imagem original para visualizar o resultado
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)

    # Exibir a máscara e a imagem mascarada
    cv2.imshow("Máscara", mask)
    cv2.imshow("Imagem com Máscara Aplicada", cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))

# Carregar a imagem
image_path = 'data/bronze/amostras/AB_00078986195518292303.png'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Criar uma janela para os sliders
cv2.namedWindow("Sliders")

# Adicionar sliders para ajustar os intervalos de RGB
cv2.createTrackbar("R Min", "Sliders", 0, 255, update_mask)
cv2.createTrackbar("R Max", "Sliders", 255, 255, update_mask)
cv2.createTrackbar("G Min", "Sliders", 0, 255, update_mask)
cv2.createTrackbar("G Max", "Sliders", 255, 255, update_mask)
cv2.createTrackbar("B Min", "Sliders", 0, 255, update_mask)
cv2.createTrackbar("B Max", "Sliders", 255, 255, update_mask)

# Inicializar as janelas de exibição
cv2.imshow("Máscara", np.zeros_like(image_rgb[:, :, 0]))  # Máscara inicial vazia
cv2.imshow("Imagem com Máscara Aplicada", image)  # Imagem original

# Esperar até que o usuário pressione 'Esc' para sair
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Pressione 'Esc' para sair
        break

cv2.destroyAllWindows()