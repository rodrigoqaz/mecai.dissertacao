import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
import random

# Caminhos dos datasets
data_v2_path = 'data/gold/dataset_224_v2/'  # Dataset V2
data_v3_path = 'data/gold/dataset_224_v4/'  # Novo dataset V3

# Garantir que o diretório V3 exista
os.makedirs(data_v3_path, exist_ok=True)

# Criar camadas de data augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),  # Flip horizontal e vertical
    RandomRotation(0.3),  
    RandomZoom(0.2),  
])

# Contar o número de imagens em cada classe no V2
class_image_counts = {}
for class_name in os.listdir(data_v2_path):
    class_dir = os.path.join(data_v2_path, class_name)
    if os.path.isdir(class_dir):
        class_image_counts[class_name] = len([img for img in os.listdir(class_dir) if img.endswith('.png')])

# Encontrar a classe com maior número de imagens
max_class_count = max(class_image_counts.values())
print(f"Classe com maior número de imagens: {max_class_count} imagens")

# Função para aplicar oversampling em classes minoritárias
def oversample_class(class_dir, output_dir, target_count):
    # Criar diretório de saída para a classe no V4
    os.makedirs(output_dir, exist_ok=True)

    # Listar todas as imagens na classe
    images = [img for img in os.listdir(class_dir) if img.endswith('.png')]

    # Copiar as imagens originais para o novo diretório
    for image_name in images:
        shutil.copy(os.path.join(class_dir, image_name), os.path.join(output_dir, image_name))

    # Número de imagens adicionais necessárias para atingir `target_count`
    additional_images_needed = target_count - len(images)

    # Gerar novas imagens usando data augmentation
    for i in range(additional_images_needed):
        # Escolher uma imagem aleatória da classe original
        random_image_name = random.choice(images)
        image_path = os.path.join(class_dir, random_image_name)
        
        # Carregar a imagem como tensor e redimensionar para (224, 224)
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_tensor = tf.convert_to_tensor(image_array)

        # Aplicar data augmentation para criar uma nova imagem
        augmented_image = data_augmentation(image_tensor[None, ...])  # Adicionar dimensão batch
        augmented_image_array = tf.squeeze(augmented_image).numpy()  # Remover dimensão batch

        # Salvar a nova imagem no diretório V3
        new_image_name = f"{os.path.splitext(random_image_name)[0]}_aug_{i}.png"
        save_img(os.path.join(output_dir, new_image_name), augmented_image_array)

# Aplicar oversampling em todas as classes do dataset V2
for class_name in os.listdir(data_v2_path):
    class_dir = os.path.join(data_v2_path, class_name)
    output_class_dir = os.path.join(data_v3_path, class_name)

    if os.path.isdir(class_dir):
        print(f"Processando oversampling na classe: {class_name}")
        oversample_class(class_dir, output_class_dir, max_class_count)

print("Criação do dataset V4 com oversampling concluída!")
