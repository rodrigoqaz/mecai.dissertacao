import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report
import optuna
import mlflow
from tqdm import tqdm
import gc
import psutil
import subprocess


class Config:
    """
    Config parameters
    """
    data_dir = 'data/gold/datasets/v1/'
    max_epochs = 50
    patience = 5
    mlflow_uri = "file:///Users/rodrigoqaz/Documents/tmp/mlruns"
    experiment_name = "VGG16 GPU - Aumento de Dados (v2)"
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    num_workers = 4
    dataset_name = "v1"
    aug = True

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


class TransformSubset(torch.utils.data.Dataset):
    """Aplica transformações dinamicamente em subsets"""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def load_datasets(batch_size):
    full_dataset = datasets.ImageFolder(root=Config.data_dir, transform=None)
    class_names = full_dataset.classes
    class_counts = torch.tensor(
        [
            len([x for x in full_dataset.targets if x == c])
            for c in range(len(full_dataset.classes))
        ],
        dtype=torch.float32
    )

    class_weights = (1. / class_counts).cpu()
    sample_weights = class_weights[full_dataset.targets].cpu().to(torch.float32)

    num_classes = len(full_dataset.classes)

    total = len(full_dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size

    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_dataset = TransformSubset(train_subset, transform=Config.train_transform)
    val_dataset = TransformSubset(val_subset, transform=Config.val_transform)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights[train_subset.indices],
        num_samples=len(train_subset),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=Config.num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, num_classes, class_weights, class_names


def initialize_model(num_classes, dropout_rate_1, dropout_rate_2, hidden_units):
    "Inicializa o modelo com os pesos treinados"

    base_model = models.vgg16(weights='IMAGENET1K_V1')
    base_model = base_model.to(Config.device, dtype=torch.float32)

    for param in base_model.features.parameters():
        param.requires_grad = False
    
    # Descongela as 4 últimas camadas convolucionais
    for layer in base_model.features[-4:]:
        layer.requires_grad_(True)  # Ativa gradientes diretamente
        for param in layer.parameters():
                param.requires_grad = True

    # for layer in base_model.features[-4:]:
    #     if isinstance(layer, nn.Conv2d):
    #         layer.to(Config.device, dtype=torch.float32)
    #         for param in layer.parameters():
    #             param.requires_grad = True

    # Substitui o classifier
    base_model.classifier = nn.Sequential(
        nn.Linear(25088, hidden_units, dtype=torch.float32),
        nn.BatchNorm1d(hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout_rate_1),
        nn.Linear(hidden_units, hidden_units // 2, dtype=torch.float32),
        nn.BatchNorm1d(hidden_units // 2),
        nn.ReLU(),
        nn.Dropout(dropout_rate_2),
        nn.Linear(hidden_units // 2, num_classes, dtype=torch.float32)
    )
    return base_model


def get_optimizer(model, learning_rate, kernel_regularizer):
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=kernel_regularizer)


def get_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=3, 
            factor=0.1
        )
    return scheduler


class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.verbose = verbose
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def restore_best_weights(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in tqdm(loader, desc="Treinando", leave=False):
        inputs = inputs.to(Config.device, non_blocking=True, dtype=torch.float32)
        labels = labels.to(Config.device, non_blocking=True, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.detach().item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def validate(model, loader, criterion, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(Config.device, dtype=torch.float32)
            labels = labels.to(Config.device, dtype=torch.long)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Cálculo do relatório de classificação
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    val_loss = total_loss / len(loader)
    val_acc = report['accuracy']
    
    return val_loss, val_acc, report


def train_model(model, train_loader, val_loader, criterion, optimizer, patience, max_epochs, scheduler, class_names):
    early_stopping = EarlyStopping(patience=patience, verbose=False)
    for epoch in range(max_epochs):
        
        # Treinamento
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)

        # Validação
        val_loss, val_acc, class_metrics = validate(model, val_loader, criterion, class_names)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log das Métricas
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        for cls in class_names:
            mlflow.log_metric(f"precision_{cls}", class_metrics[cls]['precision'], step=epoch)
            mlflow.log_metric(f"recall_{cls}", class_metrics[cls]['recall'], step=epoch)

        print(f"Treinamento Época {epoch+1} Finalizado - train_accuracy: {train_acc:.4f} | val_accuracy: {val_acc:.4f}")
        
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping na época {epoch+1}")
            break
    early_stopping.restore_best_weights(model)
    return model, epoch+1, early_stopping.best_loss


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-4, log=True)
    kernel_regularizer = trial.suggest_float("kernel_regularizer", 1e-6, 1e-4, log=True)
    dropout_rate_1 = trial.suggest_float('dropout_1', 0.3, 0.6) 
    dropout_rate_2 = trial.suggest_float('dropout_2', 0.2, 0.5) 
    hidden_units = trial.suggest_categorical('hidden_units', [128, 256, 512]) 

    train_loader, val_loader, num_classes, class_weights, class_names = load_datasets(batch_size)
    model = initialize_model(num_classes, dropout_rate_1, dropout_rate_2, hidden_units).to(Config.device)
    optimizer = get_optimizer(model, learning_rate, kernel_regularizer)
    scheduler = get_scheduler(optimizer)
    class_weights = class_weights.to(Config.device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.set_tags({
            "model_type": "VGG16",
            "transfer_learning": "True",
            "optimizer": "Adam",
            "dataset": Config.dataset_name
        })
        mlflow.log_params({
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "kernel_regularizer": kernel_regularizer,
            "dropout_rate_1": dropout_rate_1, 
            "dropout_rate_2": dropout_rate_2, 
            "hidden_units": hidden_units
        })

        model, effective_epochs, best_val_loss = train_model(
            model, train_loader, val_loader, criterion, optimizer, Config.patience, Config.max_epochs, scheduler, class_names
        )
        final_val_acc = validate(model, val_loader, criterion, class_names)[1]
        mlflow.log_metric("final_val_accuracy", final_val_acc)
        mlflow.log_params({
            "effective_epochs": effective_epochs,
            "best_val_loss": best_val_loss
        })
        return final_val_acc


def main():
    if str(Config.device) == 'mps':
        torch.set_default_dtype(torch.float32)
        torch.mps.empty_cache()
        torch.set_float32_matmul_precision('high') 

    mlflow.set_tracking_uri(Config.mlflow_uri)
    mlflow.set_experiment(Config.experiment_name)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())

    for trial_idx in range(50):
        try:
            # Executar trial com monitoramento de memória
            result = study.ask()
            objective(result)

            # Limpeza agressiva após cada trial
            torch.mps.empty_cache()
            gc.collect()
            
            # Limpar swap se necessário
            if psutil.swap_memory().percent > 50:
                subprocess.run(["sudo", "purge"], check=True)
            
        except MemoryError:
            print(f"Trial {trial_idx} interrompido por falta de memória")
            continue


    # study.optimize(objective, n_trials=50)
    

if __name__ == '__main__':
    main()
