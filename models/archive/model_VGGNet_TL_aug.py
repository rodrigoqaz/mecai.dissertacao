# Transfer Learning com a VGGNet com aumento de dados (com as classes balanceadas)

import tensorflow as tf
from keras.api.models import Model
from keras.api.applications.vgg16 import VGG16, preprocess_input
from keras.api.layers import Flatten, Dense, Dropout
from keras.api.optimizers import Adam
from keras.api.regularizers import l2
from keras.api.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from keras.api.losses import SparseCategoricalCrossentropy
import datetime


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        print('Usando GPU')

#Melhores parâmetros: {'batch_size': 64, 'learning_rate': 7.63770227502018e-05, 'kernel_regularizer': 1.8615894000199872e-05}
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
KERNEL_REGULARIZER = 0.0001
EPOCHS = 100

## Criar os datasets
data_dir = 'data/gold/dataset_224_v3/'

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=BATCH_SIZE,
)

# Ajusta cache
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).shuffle(1000).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(buffer_size=AUTOTUNE)


# Modelo VGG já treinado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base (deixar só as últimas ligadas)
for layer in base_model.layers:
    if not layer.name in ['block5_conv1', 'block5_conv2', 'block5_conv3']:
        layer.trainable = False

# Adicionar camadas personalizadas para classificação de 10 classes
x = base_model.output
x = Flatten()(x)
x = Dense(2048, activation='relu', kernel_regularizer=l2(KERNEL_REGULARIZER))(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='sigmoid', kernel_regularizer=l2(KERNEL_REGULARIZER))(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(KERNEL_REGULARIZER))(x)
x = Dropout(0.5)(x)
output_layer = Dense(10, activation='softmax')(x)

# Criar o modelo final
model = Model(inputs=base_model.input, outputs=x)

# Resumo do modelo
model.summary()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Callbacks:
checkpoint_cb = ModelCheckpoint(
    "models/weights/vgg_tl_aug_v3-5.keras",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1,
)

log_dir = "models/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_cb = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1, 
    write_graph=True
)

early_stopping_cb = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)


callbacks = [checkpoint_cb, tensorboard_cb]

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
)
