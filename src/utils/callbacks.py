import torch
import torch.nn as nn

class EarlyStopping:
    """
    Callback para interromper o treinamento quando não há melhora na métrica monitorada (ex: val_loss).
    Salva o melhor estado dos pesos do modelo durante o treinamento.
    """
    def __init__(self, patience=5, verbose=False):
        """
        Args:
            patience (int): Número de épocas sem melhora antes de parar.
            verbose (bool): Se True, imprime mensagens de progresso.
        """
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
        self.best_state = None

    def __call__(self, val_loss, model: nn.Module):
        """
        Avalia a métrica de validação e decide se deve parar o treinamento.

        Args:
            val_loss (float): Valor da loss de validação da época atual.
            model (nn.Module): Modelo PyTorch atual (para salvar os melhores pesos).
        """
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            # Salva o melhor estado dos pesos (em CPU para compatibilidade)
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model: nn.Module):
        """
        Restaura os melhores pesos salvos no modelo fornecido.

        Args:
            model (nn.Module): Modelo PyTorch no qual os pesos serão restaurados.
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.best_score = None
        self.save_best_only = save_best_only

    def __call__(self, current_score, model):
        if self.save_best_only and (self.best_score is None or current_score < self.best_score):
            self.best_score = current_score
            torch.save(model.state_dict(), self.filepath)

