import os
import tensorflow as tf
from keras.api.preprocessing.image import save_img
from keras.api.layers import RandomFlip, RandomRotation, RandomZoom

# Diretório original e novo diretório para V2
original_dataset_dir = 'data/gold/dataset_224_v1/'
v2_dataset_dir = 'data/gold/dataset_224_v2/'

# Criar camadas de data augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),  # Flip horizontal e vertical
    RandomRotation(0.2),  # Rotação aleatória até 20%
    RandomZoom(0.1),  # Zoom aleatório até 10%
])

# Função para aplicar data augmentation e salvar imagens aumentadas
def augment_and_save_images(class_dir, output_dir):
    # Criar diretório de saída para a classe, se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Iterar sobre todas as imagens na pasta da classe
    for image_name in os.listdir(class_dir):
        if image_name.endswith(".png"):  # Garantir que seja uma imagem PNG
            image_path = os.path.join(class_dir, image_name)
            
            # Carregar a imagem como tensor e redimensionar para (224, 224)
            image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_tensor = tf.convert_to_tensor(image_array)
            
            # Aplicar data augmentation várias vezes (ex.: 3 variações por imagem)
            for i in range(3):  # Número de variações por imagem
                augmented_image = data_augmentation(image_tensor[None, ...])  # Adicionar dimensão batch
                augmented_image_array = tf.squeeze(augmented_image).numpy()  # Remover dimensão batch
                
                # Salvar a imagem aumentada no novo diretório
                new_image_name = f"{os.path.splitext(image_name)[0]}_aug_{i}.png"
                save_img(os.path.join(output_dir, new_image_name), augmented_image_array)

# Iterar sobre todas as classes no diretório original
for class_name in os.listdir(original_dataset_dir):
    class_dir = os.path.join(original_dataset_dir, class_name)
    output_class_dir = os.path.join(v2_dataset_dir, class_name)

    if os.path.isdir(class_dir):  # Garantir que seja um diretório de classe
        print(f"Processando classe: {class_name}")
        augment_and_save_images(class_dir, output_class_dir)

print("Criação do dataset V2 concluída!")
