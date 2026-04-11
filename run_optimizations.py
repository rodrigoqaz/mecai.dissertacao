import subprocess

models = [
    # 'vit',
    # 'swin',
    # 'convnext',
    # 'efficientnet',
    'resnet',
    'densenet',
    'inception',
    'vgg',
]

DATA_VERSION = 'v12'
DATA_COLOR = 'BR'

for model in models:

    print(f"Rodando modelo {model}...\n")

    cmd = [
        "python",
        "optimize_model.py",
        "--model", model,
        "--study-name", f"{model}_{DATA_VERSION}_{DATA_COLOR}_opt_aug",
        "--storage", "sqlite:///optuna_dissertacao.db",
        "--objective", "mcc_loss_composite",
        "--trials", "30",
    ]

    subprocess.run(cmd, check=True)

    cmd2 = [
        "find", "./mlruns", "-type", "d", "-path", "*/artifacts/best_model",
        "-exec", "sh", "-c",
        'echo "Deletando conteúdo de: {}"; rm -rf "{}/"*',
        ";"
    ]

    subprocess.run(cmd2, check=True) 