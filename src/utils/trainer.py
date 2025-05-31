import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Tuple, Dict, Optional
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from src.config.models.config import Config


def train_epoch(
    epoch: int,
    max_epochs: int,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer
) -> Tuple[float, float]:
    """Executa uma época de treinamento"""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(loader, unit="batch") as tepoch:
        tepoch.set_description(f"Treinando Época {epoch+1}/{max_epochs}")
        for inputs, labels in tepoch:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

            _, preds = torch.max(outputs, dim=1)

            if labels.ndim == 2:  # Caso MixUp/CutMix: labels one-hot
                targets = torch.argmax(labels, dim=1)
            else:
                targets = labels

            correct += (preds == targets).sum().item()
            
            tepoch.set_postfix(
                loss=loss.item(),
                acuracia=100. * correct / total
            )

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    tqdm.write(f"Treinamento - Loss: {epoch_loss:.4f} | Acurácia: {epoch_acc:.2f}%")

    return epoch_loss, epoch_acc


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, class_names: list) -> Tuple[float, float, Dict]:
    """Executa validação completa"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    epoch_val_loss = total_loss / len(loader)
    epoch_val_acc = 100. * report['accuracy']

    tqdm.write(f"Validação   - Loss: {epoch_val_loss:.4f} | Acurácia: {epoch_val_acc:.2f}%")
    
    return epoch_val_loss, epoch_val_acc, report
