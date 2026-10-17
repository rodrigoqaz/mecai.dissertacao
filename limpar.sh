#!/bin/bash

# 1. Validação do Argumento
#    Verifica se um caminho de projeto foi passado como primeiro argumento ($1).
#    Se não, exibe uma mensagem de erro e encerra o script.
if [ -z "$1" ]; then
  echo "Erro: Nenhum diretório de projeto foi fornecido."
  echo "Uso: $0 /caminho/para/o/projeto"
  exit 1
fi

# 2. Atribuição à Variável
#    Armazena o argumento em uma variável com nome claro para facilitar a leitura.
#    As aspas duplas garantem que nomes com espaços funcionem corretamente.
PROJECT_DIR="$1"

# Valida se o diretório fornecido realmente existe.
if [ ! -d "$PROJECT_DIR" ]; then
  echo "Erro: O diretório '$PROJECT_DIR' não foi encontrado."
  exit 1
fi

echo "Iniciando limpeza no diretório: $PROJECT_DIR"

# 3. Limpeza dos Arquivos Auxiliares do LaTeX
#    Usa a variável PROJECT_DIR como base para a busca.
#    O '-maxdepth 1' evita que o find entre em subdiretórios, limpando apenas a pasta principal.
find "$PROJECT_DIR" -maxdepth 1 -type f \( \
  -name "*.aux" -o -name "*.bbl" -o -name "*.blg" -o \
  -name "*.brf" -o -name "*.fdb_latexmk" -o -name "*.fls" -o \
  -name "*.glo" -o -name "*.glsdefs" -o -name "*.idx" -o \
  -name "*.ilg" -o -name "*.ind" -o -name "*.ist" -o \
  -name "*.loa" -o -name "*.lof" -o -name "*.lol" -o \
  -name "*.loq" -o -name "*.los" -o -name "*.lot" -o \
  -name "*.log" -o -name "*.mw" -o -name "*.nlo" -o \
  -name "*.synctex.gz" -o -name "*.toc" \
\) -exec rm {} +

# 4. Limpeza de Caches do Python
#    Também usa a variável para buscar e remover pastas __pycache__.
find "$PROJECT_DIR" -type d -name "__pycache__" -exec rm -r {} +

echo "Limpeza concluída com sucesso!"