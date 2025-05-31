import os
import shutil
import random

# Caminhos dos datasets
data_v2_path = 'data/gold/dataset_224_v2/'  # Dataset V2
data_v3_path = 'data/gold/dataset_224_v3/'  # Novo dataset V3

# Garantir que o diretório V3 exista
os.makedirs(data_v3_path, exist_ok=True)

# Contar o número de imagens em cada classe no V2
class_image_counts = {}
for class_name in os.listdir(data_v2_path):
    class_dir = os.path.join(data_v2_path, class_name)
    if os.path.isdir(class_dir):
        class_image_counts[class_name] = len([img for img in os.listdir(class_dir) if img.endswith('.png')])

# Encontrar a classe com o menor número de imagens
min_class = min(class_image_counts, key=class_image_counts.get)
min_class_count = class_image_counts[min_class]
print(f"Classe com menor número de imagens: {min_class} ({min_class_count} imagens)")

# Realizar undersampling para cada classe com base na menor quantidade de imagens
for class_name in os.listdir(data_v2_path):
    class_dir = os.path.join(data_v2_path, class_name)
    output_class_dir = os.path.join(data_v3_path, class_name)

    if os.path.isdir(class_dir):
        # Criar diretório da classe no V3
        os.makedirs(output_class_dir, exist_ok=True)

        # Listar todas as imagens da classe
        images = [img for img in os.listdir(class_dir) if img.endswith('.png')]

        # Selecionar aleatoriamente `min_class_count` imagens para undersampling
        sampled_images = random.sample(images, min_class_count)

        # Copiar as imagens selecionadas para o diretório V3
        for image_name in sampled_images:
            shutil.copy(os.path.join(class_dir, image_name), os.path.join(output_class_dir, image_name))

print("Criação do dataset V3 com undersampling concluída!")
