#!/usr/bin/env bash
# Backup de data/gold/datasets/ antes do rebuild do p3.1 (re-split por fardo).
# Ver correcoes_banca/p3.1_resplit_por_fardo.md ("Riscos e cuidados").
#
# Uso (a partir da raiz do repositório):
#   ./scripts/backup_gold_datasets.sh [destino]
set -euo pipefail

SRC="data/gold/datasets"
DEST="${1:-data/gold/datasets_backup_$(date +%Y%m%d_%H%M%S)}"

if [ ! -d "$SRC" ]; then
    echo "Diretório '$SRC' não encontrado. Rode a partir da raiz do repositório." >&2
    exit 1
fi

if [ -e "$DEST" ]; then
    echo "Destino '$DEST' já existe. Escolha outro ou remova-o antes." >&2
    exit 1
fi

echo "Copiando $SRC -> $DEST ..."
cp -a "$SRC" "$DEST"
echo "Backup concluído em: $DEST"
