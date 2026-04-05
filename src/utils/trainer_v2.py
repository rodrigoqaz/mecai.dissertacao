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
    optimizer: torch.optim.Optimizer,
    device
) -> Tuple[float, float]:
    """Executa uma época de treinamento"""

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(loader, unit="batch") as tepoch:
        tepoch.set_description(f"Treinando Época {epoch+1}/{max_epochs}")
        for inputs, labels in tepoch:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # outputs, aux_outputs = model(inputs)

            model_output = model(inputs)
            if isinstance(model_output, tuple):
                outputs, *aux_outputs = model_output
            else:
                outputs = model_output
                aux_outputs = []

            # Cálculo da perda principal
            if labels.ndim == 2:  # Caso de labels modificados (ex: MixUp/CutMix)
                loss = criterion(outputs, labels.argmax(dim=1))
            else:
                loss = criterion(outputs, labels)
            
            # Adição de perdas auxiliares (se existirem)
            aux_loss_weight = 0.4  # Configurável via hyperparams
            for aux_out in aux_outputs:
                if labels.ndim == 2:
                    aux_loss = criterion(aux_out, labels.argmax(dim=1))
                else:
                    aux_loss = criterion(aux_out, labels)
                loss += aux_loss_weight * aux_loss

            # outputs = model(inputs)
            # loss = criterion(outputs, labels)
            # if labels.ndim == 2:
            #     loss = criterion(outputs, labels.argmax(dim=1)) + 0.4 * criterion(aux_outputs, labels.argmax(dim=1))
            # else:
            #     loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)
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


def validate(model: nn.Module, loader: DataLoader, criterion: nn.Module, class_names: list, device) -> Tuple[float, Dict, list, list, list]:
    """Executa validação completa"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calcula probabilidades e predições
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    epoch_val_loss = total_loss / len(loader)
    epoch_val_acc = 100. * report['accuracy']

    tqdm.write(f"Validação   - Loss: {epoch_val_loss:.4f} | Acurácia: {epoch_val_acc:.2f}%")
    
    return epoch_val_loss, report, all_labels, all_preds, all_probs
