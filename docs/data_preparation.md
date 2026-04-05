# 1. Preparação e Processamento do Conjunto de Dados

  A etapa de preparação de dados é um pilar fundamental em projetos de visão computacional, sendo decisiva para a qualidade e o desempenho do modelo final. O objetivo deste processo é transformar as imagens brutas, capturadas em ambiente controlado, em um conjunto de dados estruturado, normalizado e pronto para ser consumido por modelos de *deep learning*. Para garantir a organização, rastreabilidade e reprodutibilidade, foi adotada uma metodologia de pipeline de dados em camadas, segmentada em Bronze, Silver e Gold **[aqui acho que podemos referenciar o Databricks]**.

## 1.1. Camada Bronze: Coleta e Organização dos Dados Brutos

  A camada Bronze representa o ponto de partida do nosso fluxo de dados, contendo os arquivos em seu estado original e não processado. A coleta foi realizada por meio do script `00_download_bronze.py`, que popula o diretório `data/bronze/` com dois tipos de artefatos:

   **1. Imagens das Amostras:** Imagens em alta resolução de cada amostra de algodão, capturadas sob três condições de iluminação distintas, identificadas pelos prefixos AB (luz amarela e branca), AM (luz amarela) e BR (luz branca).
   **2. Metadados das Amostras:** Um arquivo em formato Parquet (`dados_fardinhos.parquet`), contendo informações associadas a cada amostra, incluindo seu código de identificação e, crucialmente, sua classificação de qualidade (classe), que servirá como o nosso *ground truth* (variável alvo), além de outras características de cada amostra.
   
   Esta camada funciona como um repositório de dados brutos, garantindo que os dados originais permaneçam imutáveis para futuras auditorias ou reprocessamentos.

## 1.2. Camada Silver: Segmentação e Padronização das Amostras

  A camada Silver é onde ocorre a maior parte do pré-processamento e limpeza das imagens. O objetivo é isolar o objeto de interesse (a amostra de algodão) e padronizar seu formato. Este processo foi dividido em duas fases principais.

### 1.2.1. Fase 1: Segmentação da Região de Interesse (AOI)

  A primeira fase, executada pelo script `01_create_silver_aoi_v2.py`, foca em segmentar a Região de Interesse (*Area of Interest - AOI*): o retângulo que contém exclusivamente a amostra de algodão, removendo o fundo da imagem e a etiqueta de identificação. A metodologia adotada foi a seguinte:

   **1. Imagem de Referência:** A imagem com iluminação combinada (AB) foi utilizada como referência para a segmentação, por oferecer um bom equilíbrio de cor e contraste.
   **2. Remoção de Fundo:** Utilizou-se o modelo pré-treinado `BiRefNet` para realizar a segmentação semântica e remover o fundo da imagem, isolando a amostra e a etiqueta **[Aqui deve-se referenciar corretamente o modelo e justificar a escolha de um modelo pré-treinado]**.
   **3. Mascaramento da Etiqueta:** Uma máscara binária foi criada para a etiqueta amarela por meio da segmentação por cor no espaço de cores HSV, seguida de operações morfológicas de fechamento e dilatação para garantir a cobertura completa da etiqueta.
   **4. Isolamento do Algodão:** A máscara da etiqueta foi subtraída da máscara principal (pós-remoção de fundo) para isolar unicamente a área correspondente ao algodão.
   **5. Definição da AOI:** Na máscara resultante, foi identificado o maior contorno, e a biblioteca `largest-interior-rectangle` foi empregada para calcular o maior retângulo possível contido estritamente dentro deste contorno. As coordenadas deste retângulo definem a AOI **[Explicar a fórmula para o largest interior rectangle]**.
   **6. Aplicação Consistente:** As mesmas coordenadas da AOI, calculadas a partir da imagem AB, foram aplicadas para recortar as imagens AM e BR correspondentes, garantindo que a área de análise seja idêntica entre as diferentes condições de iluminação.

  Ao final desta fase, as imagens retangulares recortadas são salvas no diretório `data/silver/amostras_recortadas/`, e um arquivo de metadados `amostras_recortadas.json` é gerado para catalogar os caminhos e as coordenadas de cada amostra.

### 1.2.2 Fase 2: Geração de Amostras Quadradas

  Modelos de redes neurais convolucionais (CNNs) tipicamente requerem imagens de entrada com dimensões fixas. Como os recortes retangulares da fase anterior possuíam tamanhos variados, foi necessário padronizá-los. Esta fase, conduzida pelo script `02_create_silver_squares.py`, implementa uma estratégia de ladrilhamento (*tiling*):

   **1. Geração de *Patches*:** Cada recorte retangular foi dividido em múltiplos *patches* (recortes) quadrados de 256x256 pixels. Uma sobreposição entre os *patches* foi utilizada para garantir a cobertura total da amostra e, como benefício secundário, aumentar o volume de dados de treinamento.
   **2. Enriquecimento dos Metadados:** O script atualizou o arquivo `amostras_recortadas.json`, adicionando a ele duas informações cruciais: as coordenadas de cada patch quadrado gerado e a classe da amostra, extraída do arquivo Parquet da camada Bronze.

  O resultado é um conjunto de imagens padronizadas em `data/silver/amostras_recortadas_quadrado/` e um arquivo JSON que serve como um catálogo completo, mapeando cada imagem quadrada à sua amostra de origem e sua respectiva classe.

## 1.3 Camada Gold: Estruturação dos Conjuntos de Dados para Treinamento

  A camada Gold representa a versão final e consolidada do conjunto de dados, pronta para o consumo direto pelos algoritmos de aprendizado de máquina. O script `03_build_gold_datasets.py` é responsável por esta última etapa.

  Utilizando as informações do arquivo JSON e arquivos de configuração que definem as diferentes versões e estratégias de divisão dos dados, o script converte as imagens quadradas de 256x256 pixels de png para matrizes numpy e as organiza em uma estrutura de diretórios hierárquica. As imagens são copiadas para as subpastas `train/`, `val/` e `test/` dentro de `data/gold/`, e, dentro de cada uma, são novamente separadas em subdiretórios nomeados de acordo com suas classes.

  Esta estrutura é o formato padrão esperado por carregadores de dados de frameworks como PyTorch e TensorFlow, permitindo que os conjuntos de treinamento, validação e teste sejam carregados de forma eficiente e direta durante a fase de experimentação. Ao final deste pipeline, o conjunto de dados está otimizado para a reprodutibilidade e a execução dos modelos de classificação.