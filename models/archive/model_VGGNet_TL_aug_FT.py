import tensorflow as tf
from keras.api.models import Model
from keras.api.applications.vgg16 import VGG16, preprocess_input
from keras.api.layers import Flatten, Dense, Dropout
from keras.api.optimizers import Adam
from keras.api.regularizers import l2
from keras.api.losses import SparseCategoricalCrossentropy
from keras.api.callbacks import EarlyStopping
import optuna
import mlflow
from optuna.integration.mlflow import MLflowCallback
# from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()
	

# Configurar MLflow
mlflow.set_tracking_uri("file:///Users/rodrigoqaz/Documents/tmp/mlruns")  
mlflow.set_experiment("VGG16 GPU Hyperparameter Optimization")

# Configurar GPU (se disponível)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Diretório dos dados
data_dir = 'data/gold/dataset_224_v4/'

# Função para carregar os datasets com aumento de dados
def load_datasets(batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=batch_size,
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (preprocess_input(x), y)).cache().prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    kernel_regularizer_value = trial.suggest_float("kernel_regularizer", 1e-5, 1e-2, log=True)

    train_ds, val_ds = load_datasets(batch_size)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Congelar todas as camadas exceto a última convolucional
    for layer in base_model.layers:
        if not layer.name in ['block5_conv3']:
            layer.trainable = False

    # Adicionar camadas personalizadas para classificação de 10 classes
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(kernel_regularizer_value))(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(kernel_regularizer_value))(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)

    model.summary()

    # Compilar o modelo com o learning rate otimizado
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # Callbacks para treinamento (EarlyStopping e MLflow autologging)
    early_stopping_cb = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    with mlflow.start_run(nested=True):  # Iniciar um run no MLflow para este trial
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("kernel_regularizer", kernel_regularizer_value)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=50,
            callbacks=[early_stopping_cb], verbose=False
        )

        # Registrar métricas por época no MLflow
        for epoch in range(len(history.history["loss"])):
            mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
            mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)
            mlflow.log_metric("train_accuracy", history.history["accuracy"][epoch], step=epoch)
            mlflow.log_metric("val_accuracy", history.history["val_accuracy"][epoch], step=epoch)

        # Registrar métricas finais no MLflow
        val_accuracy_final = max(history.history["val_accuracy"])
        mlflow.log_metric("final_val_accuracy", val_accuracy_final)

        return val_accuracy_final

# Configurar o estudo do Optuna com integração ao MLflowCallback para rastreamento automático dos trials no MLflow
mlflc = MLflowCallback(metric_name="final_val_accuracy")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=50, callbacks=[mlflc])

# Exibir os melhores parâmetros encontrados pelo Optuna após o tuning
print(f"Melhores parâmetros: {study.best_params}")
print(f"Melhor acurácia de validação: {study.best_value}")
