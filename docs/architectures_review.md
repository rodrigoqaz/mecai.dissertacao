Esta Revisão Sistemática da Literatura (RSL) visa fundamentar as escolhas arquiteturais para a classificação de 10 classes de pluma de algodão, um problema caracterizado pela alta similaridade inter-classes e pela necessidade de extração de padrões de microtextura. A análise a seguir disseca o estado da arte em visão computacional, contrastando paradigmas convolucionais e baseados em atenção.

### 1. Taxonomia das Arquiteturas: Vieses Indutivos vs. Contexto Global

A literatura contemporânea em *Deep Learning* divide-se fundamentalmente em duas vertentes arquiteturais: as Redes Neurais Convolucionais (CNNs) e os *Vision Transformers* (ViTs).

As CNNs operam sob fortes vieses indutivos (*inductive biases*), especificamente a localidade e a invariância por translação. **He et al. (2016)**, ao introduzirem a **ResNet** (*Residual Network*), resolveram o problema da degradação do gradiente em redes profundas através de conexões de atalho (*skip connections*), permitindo o aprendizado de características hierárquicas profundas essenciais para a distinção de padrões visuais complexos. Evoluções subsequentes, como a **DenseNet** proposta por **Huang et al. (2017)**, maximizaram o fluxo de informação e a reutilização de características (*feature reuse*) ao conectar cada camada a todas as subsequentes, o que se mostrou particularmente eficaz em cenários com dados limitados, embora com alto custo de memória durante o treinamento.

Em contrapartida, os modelos baseados em *Self-Attention*, inaugurados na visão pelo **ViT** (*Vision Transformer*) de **Dosovitskiy et al. (2021)**, abandonam os vieses indutivos em favor de uma modelagem de contexto global desde as primeiras camadas. No entanto, conforme observado por **Liu et al. (2022)** na proposição da **ConvNeXt**, o ViT original apresenta complexidade quadrática em relação ao número de *patches* da imagem e dificuldade em convergir sem conjuntos de dados massivos (na ordem de centenas de milhões de imagens, como o JFT-300M).

A convergência desses paradigmas ocorre com o **Swin Transformer** (*Hierarchical Vision Transformer using Shifted Windows*), proposto por **Liu et al. (2021)**. Esta arquitetura reintroduz a hierarquia e a localidade através de janelas deslizantes (*shifted windows*), reduzindo a complexidade de quadrática para linear em relação ao tamanho da imagem, tornando viável o processamento de altas resoluções necessárias para a análise de texturas finas.

### 2. Análise Crítica de Feature Extraction para Texturas de Algodão

A classificação de plumas de algodão exige a captura de detalhes de alta frequência (fibras, impurezas) e invariância a transformações espaciais.

*   **EfficientNet e Escalonamento Composto:** **Tan e Le (2019)** demonstraram que o aumento arbitrário da profundidade ou largura da rede leva a retornos decrescentes. Através do *Compound Scaling*, a **EfficientNet** equilibra profundidade, largura e resolução. Para texturas de algodão, a resolução de entrada é crítica; o método de Tan e Le garante que o campo receptivo da rede cresça em conformidade com a resolução, capturando microtexturas sem perder a semântica global.
*   **ResNet e VGG:** Embora arquiteturas como a **VGG** (**Simonyan & Zisserman, 2015**) sejam robustas extratoras de textura devido ao uso extensivo de filtros $3 \times 3$, elas sofrem com a falta de conexões globais eficientes. A **ResNet**, apesar de superior à VGG, pode apresentar um viés excessivo para formas em detrimento de texturas, a menos que treinada com aumentos de dados específicos ou modificações arquiteturais como o *Squeeze-and-Excitation* (SENet).
*   **Swin Transformer:** Para a análise de fibras, o Swin Transformer apresenta uma vantagem teórica superior. O mecanismo de atenção em janelas deslocadas permite modelar dependências de longo alcance entre fibras dispersas na imagem, algo que janelas de convolução limitadas (kernels fixos) têm dificuldade em realizar sem empilhamento excessivo de camadas. **Liang et al.** demonstraram com o **SwinIR** que esta arquitetura é excepcional para restauração de imagens e super-resolução, indicando sua capacidade superior de reconstruir e classificar texturas de alta fidelidade.

### 3. Complexidade Computacional e Viabilidade de Implementação

A viabilidade de implantação em ambientes agrícolas (borda ou nuvem) exige um balanço entre *accuracy* e GFLOPs.

*   **Eficiência de Hardware:** **Liu et al. (2022)** argumentam que, apesar da eficácia teórica dos Transformers, as CNNs modernas como a **ConvNeXt** são mais eficientes em hardware padrão (GPUs/TPUs) devido a otimizações de memória e operações convolucionais maduras. A ConvNeXt-T (Tiny) atinge acurácia comparável ao Swin-T no ImageNet com *throughput* de inferência superior.
*   **Redes Densas e Leves:** **Tan, Fu e Li (2024)** propuseram a **DLW-DenseNet** (*Differentiated Learning Weighted DenseNet*), focada em reconhecimento de texturas de tecidos. Esta variação demonstrou que mecanismos de poda (*pruning*) e pesos ajustáveis podem reduzir a redundância da DenseNet original, tornando-a viável para classificação de texturas em ambientes com recursos limitados.
*   **EfficientNet:** Permanece como o padrão de eficiência. Uma EfficientNet-B0 atinge desempenho superior a uma ResNet-50 com uma fração dos parâmetros (5.3M vs 26M) e GFLOPs (0.39B vs 4.1B), sendo a candidata ideal para dispositivos *edge* no campo.

### 4. SOTA e Benchmarking em Domínios Similares

A revisão de trabalhos em domínios correlatos (classificação de tecidos, grãos e doenças foliares) fornece *proxies* valiosos para a classificação de algodão.

*   **Resultados em Texturas:** No trabalho de **Tan et al. (2024)**, aplicado ao dataset KF9 (tecidos de malha), a **DenseNet-121** superou modelos como VGG16 e ViT, atingindo 93.2% de acurácia base, enquanto a versão proposta (DLW-DenseNet) alcançou 95.4%. Isso sugere que a reutilização de *features* da DenseNet é crítica para padrões repetitivos e sutis como os de tecidos e fibras.
*   **Classificação de Doenças (Arroz/Plantas):** Estudos comparativos indicam que arquiteturas **ResNet-50** e **InceptionV3** frequentemente atingem o estado da arte (acima de 95% de acurácia) em datasets como PlantVillage e classificação de doenças do arroz, superando modelos mais antigos como VGG devido à melhor generalização e menor *overfitting*.
*   **Funções de Perda:** O desbalanceamento de classes é um problema recorrente. A literatura aponta para a eficácia da **Focal Loss** (**Lin et al., 2017**) em cenários de detecção densa e classificação desbalanceada, penalizando exemplos difíceis de classificar mais severamente do que os fáceis. Técnicas de aumento de dados como **Mixup** (**Zhang et al., 2018**) e **CutMix** (**Yun et al., 2019**) também são citadas como essenciais para melhorar a robustez de modelos (tanto CNNs quanto Transformers) em regimes de dados limitados.

### 5. Lacunas de Pesquisa (Research Gaps)

Apesar dos avanços, a literatura apresenta lacunas específicas para a aplicação em plumas de algodão:

1.  **Dependência de Dados Massivos para Transformers:** **Dosovitskiy et al. (2021)** e **Liu et al. (2022)** confirmam que arquiteturas baseadas em atenção (ViT, Swin) sofrem de *overfitting* em datasets pequenos sem uma forte regularização ou pré-treinamento massivo (ImageNet-21k/JFT-300M). A adaptação destes modelos para datasets de algodão com anotação limitada permanece um desafio em aberto.
2.  **Robustez à Iluminação e Variação Intrínseca:** **Tan et al. (2024)** apontam que métodos baseados em estatística de textura falham sob variações de iluminação, e mesmo CNNs profundas podem ter dificuldades se não houver um pré-processamento robusto ou mecanismos de atenção adaptativa (como *Squeeze-and-Excitation*) integrados.
3.  **Similaridade Inter-classes:** A distinção entre 10 classes de pluma (que podem variar apenas ligeiramente em cor ou comprimento da fibra) exige uma análise de granulação fina (*fine-grained*) que a maioria dos *benchmarks* genéricos (ImageNet) não reflete adequadamente. A literatura carece de estudos comparativos entre **ConvNeXt** e **Swin V2** especificamente para discriminação de materiais fibrosos sob condições de captura não controladas.

Em conclusão, para a classificação de 10 classes de pluma de algodão, recomenda-se a investigação de arquiteturas híbridas ou CNNs modernizadas (**ConvNeXt** ou **EfficientNetV2**), priorizando a eficiência de parâmetros e a capacidade de extração de texturas finas, utilizando estratégias de treinamento como *Mixup* e *Focal Loss* para mitigar a escassez de dados e o desbalanceamento.



Com base na estrutura taxonômica proposta e nos documentos fornecidos, apresento a Revisão Sistemática da Literatura (RSL) focada na classificação de imagens de algodão, aprofundando-se nas especificidades arquiteturais e nos pipelines de treinamento de cada grupo.

---

### Revisão Sistemática de Arquiteturas para Classificação de Padrões Visuais em Algodão

A classificação de 10 classes de pluma de algodão apresenta desafios intrínsecos de visão computacional, notadamente a necessidade de discriminação de granulação fina (*fine-grained*) entre classes com alta similaridade visual e a captura de microtexturas fibrosas. A literatura atual pode ser categorizada em quatro grupos distintos, variando desde modelos baseados em atenção global até arquiteturas convolucionais clássicas.

#### Grupo A: Transformers (Sensibilidade e Modelagem Global)

Este grupo representa a mudança de paradigma do viés indutivo local para a modelagem de dependências globais via mecanismos de *Self-Attention*.

*   **Vision Transformer (ViT):** **Dosovitskiy et al. (2021)** demonstraram que o ViT, ao tratar imagens como sequências de *patches* (ex: $16 \times 16$), supera as CNNs em regimes de dados massivos (JFT-300M), mas sofre em *datasets* menores devido à falta de vieses indutivos como a invariância à tradução e localidade. A arquitetura exige um pipeline de treinamento robusto e "pesado", sendo altamente sensível à regularização. O uso de otimizadores como **AdamW** e técnicas de aumento de dados (Mixup, CutMix) é mandatório para evitar *overfitting* e garantir a convergência em tarefas de classificação de texturas complexas, onde a relação global entre *patches* distantes pode definir a classe da pluma.
*   **Swin Transformer:** Para mitigar a complexidade quadrática do ViT em imagens de alta resolução — essenciais para ver detalhes da fibra de algodão — **Liu et al. (2021)** introduziram o Swin Transformer. Esta arquitetura reintroduz a hierarquia através de *Shifted Windows* (janelas deslizantes), limitando a autoatenção a janelas locais não sobrepostas e permitindo a comunicação entre janelas em camadas subsequentes. Isso reduz a complexidade para linear em relação ao tamanho da imagem, tornando o modelo viável para tarefas de predição densa e classificação detalhada. A hierarquia do Swin permite a extração de características em múltiplas escalas, crucial para diferenciar impurezas locais de padrões globais da fibra.

#### Grupo B: CNNs Modernas (Eficiência e Modernização)

Este grupo engloba a resposta das redes convolucionais à ascensão dos Transformers, focando em eficiência paramétrica e adoção de técnicas modernas de treinamento.

*   **ConvNeXt:** **Liu et al. (2022)** propuseram a ConvNeXt como uma arquitetura puramente convolucional que emula o design dos Transformers (como o Swin). A "modernização" inclui o uso de *kernels* grandes ($7 \times 7$) para aumentar o campo receptivo, substituição de Batch Normalization por Layer Normalization, uso de ativações GELU e adoção de *Inverted Bottlenecks*. Crucialmente, a ConvNeXt adota o mesmo pipeline de treinamento agressivo dos Transformers (300 épocas, AdamW, Mixup, Label Smoothing), demonstrando que grande parte do ganho de performance atribuído aos Transformers advém, na verdade, dessas técnicas de treinamento avançadas. Para algodão, a ConvNeXt oferece a robustez das CNNs com a capacidade de representação dos Transformers.
*   **EfficientNet:** Focada na eficiência computacional, a EfficientNet, proposta por **Tan e Le (2019)**, utiliza o método de *Compound Scaling* para balancear profundidade, largura e resolução da rede simultaneamente. Utilizando blocos MBConv (*Mobile Inverted Bottleneck*) e mecanismos de atenção de canal (*Squeeze-and-Excitation* - SE), a EfficientNet otimiza a extração de *features* relevantes com um custo computacional significativamente menor que ResNets ou ViTs. A versão V2 (**EfficientNetV2**) aprimora a velocidade de treinamento e a eficiência paramétrica através de *Fused-MBConv* e escalonamento progressivo de imagem, ideal para ambientes de produção agrícola com recursos limitados.

#### Grupo C: CNNs Robustas (Feature Reuse e Profundidade)

Arquiteturas estabelecidas que priorizam o fluxo de gradiente e a reutilização de características, servindo como *baselines* confiáveis.

*   **ResNet:** **He et al. (2016)** revolucionaram o treinamento de redes profundas com a introdução de conexões residuais (*skip connections*), que permitem o aprendizado da função residual $F(x) = H(x) - x$ em vez do mapeamento direto, mitigando o desaparecimento do gradiente. Embora robusta, a ResNet padrão pode sofrer com redundância de características e menor eficiência paramétrica comparada às arquiteturas modernas.
*   **DenseNet:** **Huang et al. (2017)** levaram a conectividade ao extremo, conectando cada camada a todas as subsequentes (*feed-forward*). Isso maximiza o *feature reuse* (reutilização de características), permitindo que a rede opere com camadas mais estreitas e menos parâmetros que a ResNet para a mesma acurácia. Para a classificação de algodão, onde padrões de textura de baixo nível podem ser relevantes em estágios profundos da rede, a DenseNet oferece uma vantagem teórica ao manter o "conhecimento coletivo" acessível globalmente.
*   **Inception:** Caracteriza-se pelo processamento multi-escala paralelo dentro do mesmo módulo, utilizando filtros de tamanhos variados ($1 \times 1$, $3 \times 3$, $5 \times 5$) para capturar detalhes espaciais em diferentes frequências. Embora eficaz, sua complexidade de engenharia foi eventualmente superada pela simplicidade e escalabilidade das ResNets e EfficientNets.

#### Grupo D: Legacy (Uniformidade e Custo)

*   **VGG:** A arquitetura VGG, desenvolvida por **Simonyan e Zisserman**, é notável por sua simplicidade e uso exclusivo de filtros $3 \times 3$ empilhados. Apesar de sua importância histórica na demonstração de que a profundidade é crítica para a performance, a VGG é computacionalmente ineficiente devido às suas camadas densas massivas, resultando em um alto número de parâmetros e GFLOPs. Em comparações contemporâneas, a VGG geralmente apresenta desempenho inferior e maior custo de inferência do que as arquiteturas dos Grupos B e C, servindo principalmente como linha de base para demonstrar a evolução da eficiência.

### Análise Crítica do Pipeline de Treinamento

A revisão da literatura indica uma bifurcação clara nas estratégias de treinamento. Enquanto o **Grupo D (Legacy)** e partes do **Grupo C (Robustas)** eram treinados com SGD (Stochastic Gradient Descent) e aumentos de dados simples, os **Grupos A (Transformers)** e **B (Modernas)** exigem um regime de regularização estrito.

O sucesso do **ConvNeXt** e do **Swin Transformer** na classificação de texturas complexas depende intrinsecamente do uso de **AdamW** para desacoplar o decaimento de peso (*weight decay*), e de técnicas como **Stochastic Depth** (que descarta aleatoriamente camadas durante o treino) para prevenir a co-adaptação de neurônios. Para o problema de 10 classes de algodão, a adoção destas técnicas modernas é recomendada mesmo se uma arquitetura do Grupo C for escolhida, pois a literatura sugere que "receitas" de treinamento modernas podem rejuvenescer modelos clássicos como a ResNet-50, aproximando-os do estado da arte.









---

## Revisão da Literatura: Evolução das Arquiteturas para Classificação de Padrões Visuais

A classificação de plumas de algodão em 10 classes exige a captura de detalhes de alta frequência (fibras e impurezas) e a superação da alta similaridade inter-classes. A análise a seguir disseca a evolução das arquiteturas, do rigor local das CNNs à flexibilidade global dos Transformers.

### 1. Evolução Arquitetural: Da Profundidade à Eficiência

A trajetória do estado da arte em visão computacional pode ser dividida em quatro estágios evolutivos:

#### O Domínio das Redes Convolucionais (CNNs) e o Viés da Localidade

As CNNs dominam a visão computacional pelo uso de vieses indutivos estruturais: a localidade e a invariância por translação.

* **Fundamentos (ResNet):** A introdução das conexões residuais por He et al. (2016) permitiu que o modelo aprenda mapeamentos de identidade, garantindo que a extração de microtexturas de baixo nível nas plumas não seja perdida em redes extremamente profundas.

* **Eficiência e Escalonamento (EfficientNet):** Tan e Le (2019) provaram que aumentar apenas a profundidade é ineficiente. Através do Compound Scaling, a EfficientNet equilibra resolução, largura e profundidade. Para a pluma de algodão, isso é vital: o modelo garante que o campo receptivo cresça para capturar a continuidade das fibras sem desperdiçar parâmetros em redundâncias, tornando-a o padrão para eficiência em produção.

* **Estágio I: Fundamentos e Profundidade (VGG e Inception):** A **VGG (Simonyan & Zisserman, 2015)** estabeleceu o uso de filtros  empilhados para extração de textura, mas sua ineficiência paramétrica e ausência de conexões globais a tornam obsoleta para produção. Já a linha **Inception** (CITAR) introduziu o processamento multi-escala paralelo, capturando detalhes em diferentes frequências espaciais, embora com alta complexidade de engenharia.
* **Estágio II: Fluxo de Gradiente e Reuso (ResNet e DenseNet):** A introdução das conexões residuais na **ResNet (He et al., 2016)** mitigou a degradação do gradiente, permitindo redes mais profundas. A **DenseNet (Huang et al., 2017)** evoluiu este conceito através da "reutilização de características" (*feature reuse*), conectando camadas subsequentes de forma densa. Esta abordagem é particularmente eficaz para algodão, onde padrões de textura de baixo nível precisam ser preservados até as camadas finais de decisão.
* **Estágio III: Atenção Global e Transformers (ViT e Swin):** O **ViT (Dosovitskiy et al., 2021)** rompeu com os vieses indutivos (localidade) ao aplicar o mecanismo de *Self-Attention* sobre *patches* da imagem. Embora poderoso em contextos globais, sua complexidade quadrática foi resolvida pelo **Swin Transformer (Liu et al., 2021)**, que utiliza janelas deslizantes (*shifted windows*). Para a análise de fibras, o Swin permite modelar dependências de longo alcance entre fibras dispersas, superando a limitação de kernels fixos das CNNs.
* **Estágio IV: Convergência e Modernização (ConvNeXt e EfficientNet):** A **ConvNeXt (Liu et al., 2022)** "modernizou" as CNNs ao integrar elementos dos Transformers (kernels  e *Inverted Bottlenecks*), provando que redes convolucionais podem atingir o SOTA se utilizarem os mesmos protocolos de treinamento agressivos. Paralelamente, a **EfficientNet (Tan & Le, 2019)** otimizou o escalonamento composto (profundidade, largura e resolução), sendo a candidata ideal para dispositivos *edge* no campo.

### 2. Análise Crítica Aplicada às Microtexturas de Algodão

A extração de *features* em plumas de algodão exige um equilíbrio entre invariância espacial e sensibilidade ao detalhe.

* **Otimização de Textura:** Enquanto a ResNet foca em formas, o **Swin Transformer** e a **DenseNet** demonstram superioridade teórica para fibras. O SwinIR (uma variação para restauração) provou que a atenção em janelas é excepcional para reconstruir texturas de alta fidelidade.
* **Viabilidade e Implementação:** A literatura aponta um *trade-off* crítico: embora Transformers sejam superiores em precisão teórica, as CNNs modernas como a **ConvNeXt** e **EfficientNetV2** apresentam maior *throughput* em hardware padrão (GPUs) devido a operações convolucionais maduras. Para ambientes de borda, a EfficientNet-B0 mantém uma acurácia competitiva com uma fração dos parâmetros (5.3M vs 26M da ResNet-50).

### 3. Síntese do Pipeline de Treinamento e Lacunas (Research Gaps)

A performance na classificação de 10 classes depende tanto da arquitetura quanto da "receita" de treinamento.

> **Consolidação Técnica:** A adoção de otimizadores **AdamW**, funções de perda robustas como **Focal Loss** (para mitigar desbalanceamento de classes) e técnicas de aumento de dados como **Mixup** e **CutMix** é mandatória. A literatura prova que essas técnicas são responsáveis por grande parte dos ganhos atribuídos aos modelos SOTA.

**Lacunas Identificadas:**

1. **Sensibilidade ao Volume de Dados:** Transformers ainda exigem regularização extrema ou pré-treino massivo para não sofrerem *overfitting* em *datasets* agrícolas menores.
2. **Similaridade Inter-classes:** A maioria dos modelos é validada no ImageNet. A distinção de granulação fina (*fine-grained*) para fibras de cores e comprimentos similares ainda carece de estudos comparativos entre **ConvNeXt** e **Swin V2** sob condições de captura não controladas.

---

### Próximo Passo Sugerido

Gostaria que eu gerasse uma **Tabela Comparativa em formato LaTeX** consolidando os GFLOPs, Parâmetros e o "Veredito Acadêmico" de cada um desses quatro grupos para você inserir diretamente na sua dissertação?


Ainda falta:

3.3. Modernização e Eficiência Convolucional (EfficientNet e ConvNeXt)
3.4. Paradigmas Baseados em Atenção (ViT e Swin Transformer)