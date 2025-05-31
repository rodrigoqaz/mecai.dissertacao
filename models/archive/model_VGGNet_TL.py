# Transfer Learning com a VGGNet

import tensorflow as tf
from keras.api.models import Model
from keras.api.applications.vgg16 import VGG16, preprocess_input
from keras.api.layers import Flatten, Dense
from keras.api.optimizers import Adam
from keras.api.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
from keras.api.losses import SparseCategoricalCrossentropy
import datetime


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 20

## Criar os datasets
data_dir = 'data/gold/dataset_224_v1/'

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
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(buffer_size=AUTOTUNE)


# Modelo VGG já treinado
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar as camadas do modelo base
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas personalizadas para classificação de 10 classes
x = base_model.output
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dense(10, activation='softmax')(x)

# Criar o modelo final
model = Model(inputs=base_model.input, outputs=x)

# Resumo do modelo
model.summary()

# Pesos inversos de cada classe (1/Pi)
# class_weights = [
#     0.0348,
#     0.0392,
#     0.0348,
#     0.0482,
#     0.1566,
#     0.0392,
#     0.1566,
#     0.1566,
#     0.1253,
#     0.2088,
# ]

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Callbacks:
checkpoint_cb = ModelCheckpoint(
    "models/weights/vgg_tl.keras",
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
    monitor="val_loss", patience=5, restore_best_weights=True
)

def scheduler(epoch, lr):
    return lr * 0.1 if epoch > 5 else lr

lr_scheduler_cb = LearningRateScheduler(scheduler)

callbacks = [checkpoint_cb, tensorboard_cb, lr_scheduler_cb, early_stopping_cb]

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks
)
