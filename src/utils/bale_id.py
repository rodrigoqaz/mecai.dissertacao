import re

# Nomes de patch seguem o padrão "{LUZ}_{ID do fardo, 20 dígitos}_{idx}[...].{ext}",
# ex.: "AB_00078986195518321249_8.npy" ou, após feature engineering (Gabor),
# "AB_00078986195518321249_8_threshold+gabor.npy".
BALE_ID_PATTERN = re.compile(r'^[A-Z]+_(\d{20})_\d+')


def extract_bale_id(filename: str) -> str:
    """
    Extrai o ID do fardo físico (20 dígitos) do nome de um arquivo de patch.

    Usado pelo split agrupado por fardo (p3.1) para garantir que nenhum patch de
    um fardo presente no teste apareça no treino/validação.
    """
    match = BALE_ID_PATTERN.match(filename)
    if not match:
        raise ValueError(
            f"Não foi possível extrair o ID do fardo do nome de arquivo: '{filename}'. "
            "Esperado o formato '{LUZ}_{ID de 20 dígitos}_{idx}...'."
        )
    return match.group(1)
