#!/usr/bin/env python3
"""
Script principal para otimização de hiperparâmetros usando Optuna.
Executa N trials para um modelo específico, integrando com MLflow.
"""

import argparse
import optuna
from src.optimization.objective import OptimizationObjective
from src.config.models.config import Config
import os


def main():
    parser = argparse.ArgumentParser(description='Otimização de hiperparâmetros com Optuna')
    parser.add_argument('--model', '-m', type=str, required=True,
                       choices=['resnet', 'inception', 'efficientnet', 'densenet', 'vgg', 'vit', 'swin', 'convnext'],
                       help='Nome do modelo para otimizar')
    parser.add_argument('--trials', '-t', type=int, default=50,
                       help='Número de trials para executar (default: 50)')
    parser.add_argument('--study-name', '-s', type=str, default=None,
                       help='Nome do estudo (default: {model}_optimization)')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout em segundos (default: sem limite)')
    parser.add_argument('--objective', type=str, default='mcc_loss_composite',
                       choices=['accuracy', 'balanced', 'loss', 'mcc', 'mcc_loss_composite'],
                       help='Tipo de objetivo a otimizar (default: mcc_loss_composite)')
    parser.add_argument('--storage', type=str, default='sqlite:///optuna_study.db',
                       help='URL de armazenamento do estudo Optuna (default: sqlite:///optuna_study.db)')
    
    args = parser.parse_args()
    
    # Nome do estudo
    study_name = args.study_name or f"{args.model}_optimization"
    
    # Carrega configuração base do modelo
    print(f"Carregando configuração base para {args.model}...")
    base_config = Config.load_model_config(args.model)
    
    # Cria estudo Optuna
    print(f"Criando estudo de otimização: {study_name}")
    study = optuna.create_study(
        direction="maximize",  # Maximizar accuracy/objetivo
        storage=args.storage,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        load_if_exists=True
    )
    
    # Cria função objetivo
    objective_func = OptimizationObjective(
        model_name=args.model,
        base_config=base_config,
        objective_type=args.objective,
        study_name=study_name
    )
    
    print(f"Iniciando otimização com {args.trials} trials...")
    print(f"Modelo: {args.model}")
    print(f"Objetivo: {args.objective}")
    print(f"MLflow URI: {base_config.mlflow_uri}")
    print(f"MLflow Directory: ./mlruns")
    print("-" * 50)
    
    # Executa otimização
    study.optimize(
        objective_func,
        n_trials=args.trials,
        timeout=args.timeout
    )
    
    # Resultados finais
    print("\n" + "=" * 50)
    print("OTIMIZAÇÃO CONCLUÍDA")
    print("=" * 50)
    print(f"Número de trials completados: {len(study.trials)}")
    print(f"Melhor valor: {study.best_value:.4f}")
    print(f"Melhores parâmetros:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Salvar a melhor configuração encontrada
    optimized_dir = "src/config/models/optimized"
    os.makedirs(optimized_dir, exist_ok=True)
    best_config_file = os.path.join(optimized_dir, f"best_config_{args.model}_{study_name}.yaml")
    objective_func.save_best_config(study.best_params, best_config_file)
    print(f"\nMelhor configuração salva em: {best_config_file}")


if __name__ == "__main__":
    main()
