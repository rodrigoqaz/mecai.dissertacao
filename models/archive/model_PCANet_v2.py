import numpy as np
from sklearn.decomposition import IncrementalPCA
from scipy.signal import convolve2d
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import cv2
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def normalize_image(image):
    """
    Normaliza uma imagem RGB para o intervalo [0, 1].
    
    Parâmetros:
    image: Array numpy de formato (altura, largura, 3)
    
    Retorna:
    imagem normalizada como um array numpy de floats no intervalo [0, 1]
    """
    # Verifica se a imagem tem 3 canais
    if image.shape[2] != 3:
        raise ValueError("A imagem deve ter 3 canais (RGB)")
    
    # Converte para float para evitar overflow
    image_float = image.astype(float)
    
    # Normaliza cada canal separadamente
    for channel in range(3):
        channel_min = np.min(image_float[:,:,channel])
        channel_max = np.max(image_float[:,:,channel])
        
        # Evita divisão por zero
        if channel_min != channel_max:
            image_float[:,:,channel] = (image_float[:,:,channel] - channel_min) / (channel_max - channel_min)
        else:
            image_float[:,:,channel] = image_float[:,:,channel] - channel_min
    
    return image_float


def extrair_amostras_balanceadas(X, labels, tamanho_amostra=182):
    classes = np.unique(labels)
    amostras_balanceadas_X = []
    amostras_balanceadas_labels = []

    for classe in classes:
        indices_classe = np.where(labels == classe)[0]
        if len(indices_classe) >= tamanho_amostra:
            indices_selecionados = np.random.choice(indices_classe, tamanho_amostra, replace=False)
        else:
            indices_selecionados = np.random.choice(indices_classe, tamanho_amostra, replace=True)
        
        amostras_balanceadas_X.append(X[indices_selecionados])
        amostras_balanceadas_labels.append(labels[indices_selecionados])

    return np.concatenate(amostras_balanceadas_X), np.concatenate(amostras_balanceadas_labels)


def convolution_2d(img_c, filters_c, stride=(1,1)):

    # Extrair dimensões
    n_filters=1
    n_channels=3
    print(filters_c.shape)
    f_height, f_width = filters_c.shape
    stride_h, stride_w = stride
    
    # Verificar consistência de canais
    if n_channels != img_c.shape[2]:
        raise ValueError("Número de canais nos filtros e imagens deve ser igual")

    # Calcular dimensões de saída
    sample_height, sample_width = img_c.shape[0], img_c.shape[1]
    conv_height = sample_height - f_height + 1
    conv_width = sample_width - f_width + 1
    
    # Aplicar stride
    out_height = (conv_height + stride_h - 1) // stride_h
    out_width = (conv_width + stride_w - 1) // stride_w
    
    # Inicializar mapa de características
    feature_map = np.zeros((img_c.shape[0], n_filters, out_height, out_width))

    channel_sum = np.zeros((conv_height, conv_width))
    
    for c in range(n_channels): # Loop nos canais
        channel_sum += convolve2d(
            img_c[:, :, c],
            filters_c,
            mode='valid'
        )
    
    # Aplicar stride e armazenar resultado
    feature_map = channel_sum[::stride_h, ::stride_w]
    
    return feature_map 


class PCANet:
    def __init__(self, filter_size, num_filters, num_stages, block_size):
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_stages = num_stages
        self.block_size = block_size
        self.pca_filters = []

    def fit(self, images):
        for stage in range(self.num_stages):
            pca = IncrementalPCA(n_components=self.num_filters)
            patches = self._extract_patches(images)
            pca.fit(patches)
            self.pca_filters.append(pca.components_.reshape((-1, self.filter_size, self.filter_size)))
            
            if stage < self.num_stages - 1:
                images = self._apply_filters(images, self.pca_filters[-1])

    def transform(self, images):
        for stage, filters in enumerate(self.pca_filters):
            if stage == self.num_stages - 1:
                binary_outputs = self._apply_filters(images, filters, binary=True)
                features = self._extract_histograms(binary_outputs)
                return features
            else:
                images = self._apply_filters(images, filters)

    def _extract_patches(self, images):
        patches = []
        for img in images:
            for i in range(img.shape[0] - self.filter_size + 1):
                for j in range(img.shape[1] - self.filter_size + 1):
                    patch = img[i:i+self.filter_size, j:j+self.filter_size].flatten()
                    patches.append(patch - np.mean(patch))
        return np.array(patches)

    def _apply_filters(self, images, filters, binary=False):
        outputs = []
        for img in images:
            img_outputs = []
            for filter in filters:
                print(filter.shape)
                print(img.shape)

                output = convolution_2d(img_c = img, filters_c = filter)
                if binary:
                    output = (output > 0).astype(int)
                img_outputs.append(output)
            outputs.append(img_outputs)
        return np.array(outputs)

    def _extract_histograms(self, binary_outputs):
        features = []
        for img_outputs in binary_outputs:
            img_features = []
            for i in range(0, img_outputs[0].shape[0], self.block_size[0]):
                for j in range(0, img_outputs[0].shape[1], self.block_size[1]):
                    block = [out[i:i+self.block_size[0], j:j+self.block_size[1]] for out in img_outputs]
                    block_features = self._compute_histogram(block)
                    img_features.extend(block_features)
            features.append(img_features)
        return np.array(features)

    def _compute_histogram(self, block):
        decimal = sum([2**i * block[i] for i in range(len(block))])
        hist, _ = np.histogram(decimal, bins=2**len(block), range=(0, 2**len(block)))
        return hist



ini = time.time()

print('Abrindo o arquivo')
with open('data/silver/amostras_recortadas.json', 'r') as f:
    amostras_recortadas = json.load(f)


# Cria as features:
print('Criando as features')
path_imagens = 'data/silver/amostras_recortadas_quadrado/'
labels = []
images = []
for amostra in amostras_recortadas:
    codigo = amostra['codigo']
    classe = amostra['classe']
    qtd_images = amostra['atributos']['quadrados']['qtd']
    for i in range(1, qtd_images):
        path_imagem = f"{path_imagens}AB_{codigo}_{i}.png"
        img = cv2.imread(path_imagem)
        img_norm = normalize_image(img)
        images.append(img)
        labels.append(classe)

images = np.array(images)
labels = np.array(labels)

images, labels = extrair_amostras_balanceadas(images, labels)

vals, counts = np.unique(labels, return_counts=True)

print("Valores Únicos:", vals)
print("Contagens:", counts)

print('Treinando')
# Inicializar e treinar o PCANet
pcanet = PCANet(filter_size=2, num_filters=2, num_stages=2, block_size=(2, 2))
pcanet.fit(images)
print('Transformando')
features = pcanet.transform(images)

onehot = OneHotEncoder()
labels = onehot.fit_transform(labels.reshape(-1, 1))

enc = LabelEncoder()
labels = enc.fit_transform(labels)

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Treinar o modelo SVM
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)

# Avaliar o modelo
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {accuracy:.2f}")

fim = time.time()
tempo = fim - ini
print(f"Tempo de execução: {format_time(tempo)} segundos")