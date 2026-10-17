

### 4.1 Fase 1: Benchmark Global e Paradigma Arquitetural ($H_1$)

**O que o texto vai argumentar:**
Aqui nós congelamos o *dataset* no cenário ruidoso (Luz Misturada AB + RGB). O texto vai introduzir a tabela principal e discutir como as redes modernas de alta eficiência superaram os *baselines* clássicos. É o momento de provar que a atenção local/convolução da ConvNeXt ou DenseNet superou a atenção global pura dos *Transformers* para texturas de algodão, rejeitando a $H_{0,1}$.

**Artefatos gerados pelo seu script Python:**
1.  **Tabela Principal (DataFrame para LaTeX):** Uma tabela contendo as colunas: `Modelo`, `MCC`, `F1-Score Macro`, `Kappa` e `GFLOPs`. O *script* deve filtrar **apenas** as execuções feitas com Luz AB + RGB. O melhor valor de cada coluna deve vir com a tag `\textbf{}`.
2.  **Gráfico de Barras/Boxplot (CNNs vs Transformers):** Aquele seu gráfico `h1_paradigm_comparison.pdf`. O Python deve plotar a média de MCC da família CNN contra a família *Transformer* (apenas no *dataset* AB), mostrando a barra de erro ou desvio padrão.

---

### 4.2 Análise Qualitativa e Significância Interclasses


Esta tabela apresenta as métricas de avaliação (Precisão, Revocação e Escore F1) de um modelo de classificação para diferentes categorias (os rótulos numéricos como 31.2, 31.3, etc.). 

### 1. Destaques Positivos (Melhores Desempenhos)
As classes com o melhor equilíbrio geral são aquelas com os maiores **Escores F1** (que é a média harmônica entre a precisão e a revocação):
* **Classe 44.4:** É o modelo de maior sucesso geral, com o maior F1 (0.732). Apresenta uma precisão muito alta (0.769) e uma boa revocação (0.698).
* **Classe 41.5:** Tem um desempenho excelente e curiosamente perfeitamente equilibrado, onde Precisão, Revocação e F1 são idênticos (0.724).
* **Classe 31.2:** Merece destaque por ter a **maior Revocação da tabela (0.744)**. Isso significa que o modelo consegue encontrar quase todos os exemplos reais dessa classe. O trade-off é que ele comete alguns falsos positivos para garantir isso (precisão menor, de 0.628).

### 2. Destaques Negativos (Dificuldades do Modelo)
O modelo tem dificuldade em identificar corretamente algumas classes, sugerindo sobreposição de características (classes difíceis de distinguir) ou falta de dados de treinamento:
* **Classe 31.3:** Apresenta o pior desempenho geral da tabela, com o menor F1 (0.489), evidenciado por uma revocação muito baixa (0.455) e precisão baixa (0.529).
* **Classe 31.4:** Tem a **pior Precisão da tabela (0.492)**. Isso indica que, quando o modelo prevê que um dado pertence à classe 31.4, ele erra na maioria das vezes (muitos falsos positivos).

### 3. Casos de Desequilíbrio (Alta Precisão vs. Baixa Revocação)
Algumas classes mostram que o comportamento do modelo é extremamente conservador. Ele só classifica algo nessas categorias quando tem "certeza absoluta", o que evita falsos positivos, mas gera muitos falsos negativos:
* **Classe 32.4:** É o caso mais extremo de desequilíbrio. Possui a **maior Precisão de todas (0.778)**, mas a **pior Revocação (0.406)**. O modelo raramente classifica algo incorretamente como 32.4, mas ele deixa passar mais da metade dos exemplos reais dessa classe.
* **Classes 42.4 e 41.4:** Seguem um comportamento similar, com boa precisão (0.673 e 0.619) contra uma revocação ruim (0.434 e 0.441).

A inclusão da matriz de confusão enriquece imensamente a sua análise. Enquanto a tabela anterior nos dizia *onde* o modelo estava falhando (baixa precisão ou revocação), a matriz da EfficientNet (V1) nos revela exatamente *como* e *por que* essas falhas estão ocorrendo. 

Quando analisamos o comportamento dessa arquitetura CNN, fica claro que a rede está sofrendo com sobreposição de características visuais específicas. Podemos extrair três grandes achados estruturais para a sua **Seção 4.2**:

### 1. O Efeito "Buraco Negro" de Algumas Classes (Falsos Positivos)
A matriz explica perfeitamente por que a **classe 31.4** teve a pior precisão da tabela anterior (0.492). Olhe para a coluna predita "31.4": ela atrai muitos erros de outras categorias.
* **36%** dos exemplos reais da classe 41.4 são classificados erroneamente como 31.4.
* **25%** da classe 31.3 caem aqui.
* **19%** da classe 32.4 caem aqui.
* **Conclusão para o texto:** A rede está usando a classe 31.4 como um "palpite seguro" quando encontra padrões visuais ambíguos. As características espaciais extraídas pela EfficientNet para a classe 31.4 não são exclusivas o suficiente, engolindo amostras de outras classes.

### 2. Confusões Intra-Grupo vs. Inter-Grupo
Se assumirmos que os rótulos seguem uma taxonomia hierárquica (onde o prefixo "31", "32" indica um macrogrupo de qualidade ou tipologia), a matriz revela um comportamento interessante:
* **Confusão de granularidade fina (Intra-grupo):** A classe 31.3 tem muita dificuldade de se distinguir das suas "vizinhas". Quando o modelo erra a 31.3, ele chuta 31.4 (25%) ou 31.2 (14%). O modelo entende que é da "família 31", mas a rede convolucional não consegue capturar o microdetalhe que separa o .2, .3 e .4.
* **Saltos semânticos (Inter-grupo):** Há erros graves que cruzam categorias distantes. O caso mais crítico é a **classe 44.4**, que tem um acerto razoável (70%), mas quando erra, joga massivamente **30% das amostras para a classe 33.3**. Isso sugere que, visualmente, há um padrão macro na EfficientNet que torna a classe 44.4 e a 33.3 quase idênticas sob certas condições de iluminação, textura ou ângulo.

### 3. O Paradoxo do Conservadorismo (Falsos Negativos)
Na tabela, vimos que a **classe 32.4** tinha a melhor precisão, mas a pior revocação. A matriz (linha 32.4) mostra o motivo: o modelo só acerta **41%** das vezes. Os outros 59% da classe real 32.4 estão espalhados, principalmente vazando para 33.3 (26%) e 31.4 (19%). O modelo raramente chuta 32.4 por engano, mas falha gravemente em reconhecer as características intrínsecas dessa classe quando ela de fato aparece.







**O que o texto vai argumentar:**
Você fará um mergulho profundo nos melhores modelos da Fase 1 (ex: DenseNet e ConvNeXt). O texto explicará onde o classificador se confunde (geralmente entre as classes limítrofes do MAPA, como 41.4 e 41.5) e usará o p-value do McNemar para cravar quem é o vencedor matemático do *benchmark*.

**Artefatos gerados pelo seu script Python:**
1.  **Matriz de Confusão Normalizada (Heatmap Seaborn):** Um mapa de calor limpo, paleta "Blues", do modelo campeão na Luz AM + RGB (`best_model_confusion_matrix.pdf`). O *script* deve colocar os rótulos reais no eixo Y e as predições no eixo X.
2.  **Matriz de McNemar (Heatmap Triangular):** Aquele seu gráfico de `p_values_v10.pdf`, mas gerado para os dados da Fase 1. O Python deve calcular o p-value pareado entre os modelos top 5 e pintar de cinza (ou colocar "ns") o que não for estatisticamente significante ($p > 0,05$).

---

### 4.3 Fase 2: Impacto Fotométrico e Sinal Físico ($H_2$)

**O que o texto vai argumentar:**
Aqui você tira o foco das linhas de código e volta para a caixa de captura física. O texto argumentará que a otimização da luz (AB para AM para BR) eleva a performance de todo mundo, rejeitando a $H_{0,2}$. Você usa o Teste de Friedman para dar peso estatístico a essa afirmação, mostrando que a física importou tanto quanto a matemática.

**Artefatos gerados pelo seu script Python:**
1.  **Gráfico de Linha de Evolução (`h2_lighting_impact.pdf`):** Eixo X são as luzes (AB, AM, BR). Eixo Y é o MCC. Cada linha é uma arquitetura. O *script* vai mostrar o agrupamento das linhas subindo à medida que a luz melhora para 6000K (Luz Branca).
2.  **Tabela de Postos de Friedman:** O Python extrai o *ranking* médio de cada versão de luz rodando o `scipy.stats.friedmanchisquare`.
3.  **Diagrama de Diferença Crítica (CD):** O *script* gera o diagrama provando que a Luz Branca pertence a um grupo de significância isolado na liderança (`critical_difference_friedman.pdf`).

---

### 4.4 Fase 3: Expansão Espectral e Engenharia de Características ($H_3$)

**O que o texto vai argumentar:**
Aqui entra a desconstrução do filtro de Gabor. O texto vai argumentar que as redes treinadas no *dataset* de 15 canais tiveram uma degradação severa em relação ao RGB puro, falhando em rejeitar $H_{0,3}$. A discussão girará em torno do conceito de "ruído paramétrico" e de como redes profundas preferem descobrir suas próprias *features* diretamente do RGB a receberem *features* "engessadas" classicamente.

**Artefatos gerados pelo seu script Python:**
1.  **Gráfico de Queda (Barplot Agrupado):** Eixo X com os modelos campeões. Duas barras lado a lado para cada modelo: uma barra para o MCC em **RGB** e uma barra menor/caída para o MCC em **15 Canais Gabor** (`h3_spectral_impact.pdf`). Isso ilustrará perfeitamente o impacto negativo.

---

### 4.5 Viabilidade Industrial e Trade-off Computacional ($H_4$)

**O que o texto vai argumentar:**
O grande fechamento. A DenseNet pode até ter ganhado no MCC puro, mas é pesada demais para o chão de fábrica. O texto coroará a ConvNeXt como a arquitetura suprema para o seu domínio (o *Sweet Spot*), por entregar quase a mesma acurácia com uma fração da latência, validando a $H_4$ e entregando valor prático para a parceira agroindustrial.

**Artefatos gerados pelo seu script Python:**
1.  **Gráfico de Bolhas (`efficiency_tradeoff_full.pdf`):** Eixo Y é o MCC. Eixo X é a Latência/Tempo de Inferência (usar escala logarítmica, dependendo dos *outliers* de tempo). O tamanho da bolha é o número de parâmetros ou GFLOPs. As bolhas devem ser coloridas por família (CNN Clássica, CNN Moderna, Transformer).
2.  **Tabela de Ranking Industrial:** Uma tabela sumarizando o modelo, MCC, Tamanho do Modelo (MB) e Tempo de Inferência (ms/imagem), ideal para sistemas computacionais de borda.

---

Ao padronizar o seu *script* para exportar os PDFs vetoriais seguindo estritamente esse mapa mental de 5 subseções, a escrita do seu texto fluirá quase que automaticamente em torno das imagens geradas. 

Quais das rotinas do seu Python você acha que precisarão de mais ajustes para se adequarem a essa extração cirúrgica dos dados do MLflow?






Como o seu orientador técnico virtual, eu diria que o conjunto que mapeamos até agora já garante a sua aprovação com louvor. Ele está lógico, matemático e focado no problema da indústria.

Porém, se o seu objetivo é arrancar elogios da banca e deixar a sua dissertação com cara de artigo publicado em revista de altíssimo impacto (como a *IEEE Transactions*), existem **três artefatos visuais** que costumam ser a "cereja do bolo" em teses de Visão Computacional. 

Como você já tem o *script* Python puxando tudo do MLflow, gerar esses gráficos extras custaria poucas linhas de código. Veja se algum deles faz sentido para você:

### 1. Mapas de Ativação Visual (Explainable AI - Grad-CAM)
* **Onde entraria:** Na subseção **4.2 (Análise Qualitativa)**.
* **O que é:** O famoso mapa de calor sobre a imagem original, mostrando para onde a rede neural estava "olhando" quando tomou a decisão.
* **Por que a banca ama:** Você disse que as CNNs ganharam porque focam em características locais (texturas finas) e que o Filtro de Gabor não ajudou. Se você gera um Grad-CAM da **ConvNeXt** olhando para um algodão e mostra o mapa de calor "acendendo" exatamente em cima de um nó de fibra (*nep*) ou de uma folha amarelada, você **prova visualmente** que a rede aprendeu a física do problema, e não apenas decorou *pixels*.
* **No Python:** O pacote `pytorch-grad-cam` gera isso em 5 linhas de código.

### 2. Curvas de Aprendizado (Learning Curves)
* **Onde entraria:** Pode ser no começo da **Fase 1** ou em um Apêndice.
* **O que é:** O gráfico clássico de Épocas (Eixo X) vs. Loss/MCC (Eixo Y), mostrando a linha de Treino e a linha de Validação juntas.
* **Por que a banca ama:** Uma pergunta clássica de defesa é: *"Como você garante que essa sua ConvNeXt gigante não sofreu overfitting nesses 7 mil patches?"*. O argumento verbal é bom, mas mostrar um gráfico onde a curva de validação acompanha a de treino de forma suave (sem aquele "bico" para cima no Loss de validação no final) é a prova irrefutável de que a sua técnica de *Data Augmentation* e *Dropout* do Optuna funcionou perfeitamente.

### 3. O Grid de "Acertos e Erros" Visuais
* **Onde entraria:** Na subseção **4.2 (Análise Qualitativa)**, logo após a Matriz de Confusão.
* **O que é:** Uma imagem montada com um *grid* de $3 \times 3$ ou $4 \times 4$ *patches* reais de algodão. Em cima de cada imagem você coloca: `Real: 31.2 | Predito: 31.3`.
* **Por que a banca ama:** Em Visão Computacional, nós falamos muito de tensores e FLOPs, mas o avaliador quer ver o **algodão**. Mostrar alguns exemplos visuais de onde a sua ConvNeXt errou ajuda o avaliador a ter empatia. Ele vai olhar para a foto do algodão e pensar: *"Nossa, mas esse 31.2 é tão sujo que eu também acharia que é um 31.3. A rede errou por pouco"*. Isso tira a culpa da matemática e mostra a dificuldade inerente da safra.

### 4. Curvas ROC Multiclasse (One-vs-Rest)
* **Onde entraria:** Junto com a tabela da Fase 1 ou na subseção 4.2.
* **O que é:** Já que você descreveu brilhantemente o cálculo da *Macro ROC AUC* no seu Capítulo 3 (Metodologia), a banca pode sentir falta de ver a curva desenhada. 
* **Por que a banca ama:** Mostrar a curva ROC do modelo campeão para as 10 classes ilustra visualmente a capacidade de separação da rede, independentemente do *threshold*.
* **No Python:** O `scikit-learn` tem a função `RocCurveDisplay.from_predictions` que plota isso maravilhosamente.

---

**Resumo da Ópera:**
O seu desenho experimental (Friedman, McNemar, Bolhas de Trade-off) cobre **100% da métrica e da engenharia**. 
Se você adicionar o **Grad-CAM** e o **Grid Visual**, você cobrirá **100% da intuição e da explicabilidade**.

O que você acha? Vale a pena incluir algum desses no seu *script* Python para enriquecer o Capítulo 4, ou prefere focar em rodar as matrizes estritas primeiro e deixar esses como plano B?