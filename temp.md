
CAPÍTULO 1 
INTRODUÇÃO 

O algodão (Gossypium hirsutum L.) representa uma das culturas de maior relevância econômica mundial, sendo conhecido como "ouro branco" devido ao seu impacto estratégico no agronegócio global.  A fibra do algodão sustenta uma indústria têxtil avaliada em aproximadamente 600 bilhões de dólares anuais (KHAN et al., 2020), consolidando sua posição como commodity essencial para a economia mundial.  No contexto brasileiro, o país figura entre os principais produtores globais, com produção anual de aproximadamente 2,5 milhões de toneladas de pluma em área cultivada de 1,6 milhão de hectares (CONAB, 2024), o que reforça o seu papel fundamental na balança comercial nacional. 

A classificação da qualidade das fibras constitui uma etapa crítica na cadeia produtiva e comercial do algodão, determinando diretamente o valor de mercado e o poder de negociação do produto (YAŞAR, 2021). O processo segue padrões internacionais que orientam a comercialização desde 1906, quando o Departamento de Agricultura dos Estados Unidos (USDA) criou o sistema de padronização que permanece como referência mundial.  Atualmente o USDA mantém 15 categorias de classificação de algodão, que são atualizadas anualmente e servem como referência para as classificações visuais realizadas em diversos países (Cotton Incorporated, 2017).  No Brasil, este processo é regulamentado pela Instrução Normativa n. 24/2016 do Ministério de Estado da Agricultura, Pecuária e Abastecimento (MAPA), que traça 

---

### **PÁGINA 34**

32 
Capítulo 1. Introdução 

padrões para a classificação, com requisitos de identidade e qualidade, amostragem e apresentação do produto (MAPΡΑ, 2016).  No processo produtivo do algodão a classificação é feita por amostragem, e pode ser realizada de duas formas: por método visual ou por método instrumental.  A classificação visual é realizada por especialistas com base nos Padrões Físicos Universais, levando em conta a cor das fibras, a presença de impurezas (como folhas), as contaminações de matérias estranhas e o modo de preparação (beneficiamento) do produto.  Apesar de ser uma abordagem mais ágil, ela incorpora um grau elevado de subjetividade inerente à interpretação humana, o que pode resultar em variabilidade nos resultados e suscitar questionamentos durante negociações comerciais.  Já a classificação instrumental é realizada por equipamento do tipo HVI (High Volume Instrument) e proporciona maior objetividade e reprodutibilidade.  No entanto, essa técnica exige um tempo de análise consideravelmente maior, custos operacionais elevados, preparo específico das amostras e condições laboratoriais controladas (OLIVEIRA, 2022) (MAPA, 2016). 

O principal gargalo logístico da cadeia produtiva ocorre na etapa de "emblocamento", que consiste na criação de blocos de algodão de aproximadamente 25 mil toneladas, compostos por fardos que possuem a mesma classificação (geralmente entre 110 e 125 fardos), sendo cada bloco é a menor unidade de comercialização.  A dinâmica operacional das algodoeiras, funcionando continuamente 24 horas por dia durante a safra, exige velocidade nesse processo de "emblocamento".  Por isso, as empresas optam por realizar esta etapa do processo de acordo com a classificação visual, já que os resultados da classificação por instrumentos podem demorar até 24 horas para chegar.  Mesmo com a classificação visual, que é mais rápida que a por instrumentos, existe um ponto de lentidão no processo, pois é necessário que as amostras sejam enviadas para as unidades de classificação e somente depois que o emblocamento é realizado.  Ter as unidades de classificação dentro da indústria não se mostra viável pela necessidade de se ter um ambiente controlado para garantia da integridade das amostras.  Nesse cenário, um sistema automatizado que possa ser instalado diretamente na linha de produção, capaz de analisar cada fardo em tempo real, emerge como uma solução de alto impacto para a otimização do processo.  Para enfrentar este desafio, a Visão Computacional se apresenta como a tecnologia mais adequada. 

---

### **PÁGINA 35**

33 

Trata-se da ciência responsável pela visão de uma máquina, pela forma como um computador enxerga o meio à sua volta, extraindo informações significativas a partir de imagens capturadas por diversos dispositivos como câmeras, sensores, scanners, etc. Estas informações permitem reconhecer, manipular e pensar sobre os objetos que compõem uma imagem (BALLARD; BROWN, 1982).  Desde os primeiros estudos na área, na década de 1970, até os dias atuais, o campo evoluiu drasticamente, principalmente com o advento da Inteligência Artificial, permitindo que computadores executem tarefas visuais complexas com precisão próxima ou superior à humana.  Assim, a visão computacional fornece ao computador uma infinidade de informações precisas a partir de imagens e vídeos, de forma que o computador consiga executar tarefas inteligentes, simulando e aproximando-se da inteligência humana (OLIVEIRA; HONORATO, 2024). 

Um sistema básico de visão computacional consiste em quatro características: iluminação, câmera, computador e software de processamento de imagens.  Esse sistema vai permitir a captura da imagem, sua conversão digital para armazenamento e processamento no computador e o software de processamento de imagens será responsável pelo aprimoramento, segmentação, detecção ou classificação da imagem (ZHANG; LI, 2014).  O sistema de iluminação deve ser capaz de destacar as características de interesse do objeto, em detrimento das demais características.  $\hat{E}$ um sistema de extrema importância para se garantir a eficiência e a acurácia de um sistema de visão computacional, além de ter uma grande influência na qualidade das imagens.  As chaves de um bom sistema de iluminação são aquelas que provêm uma iluminação uniforme com um espectro de características e que produzem imagens de alta qualidade (THOMASSON; SHEARER; BYLER, 2005).  A câmera é o componente principal do sistema de visão computacional, que é usado para a aquisição da imagem e para a troca de dados com o computador.  Existem diferentes tipos de dispositivos que capturam imagens em diversos espectros como monocromáticas, RGB, ultrassom, raios-x, NIR, etc (BROSNAN; SUN, 2002).  A automação da classificação visual por meio da visão computacional visa, portanto, substituir um processo manual e subjetivo por um sistema objetivo, rápido e padronizado, mitigando as incertezas que geram disputas comerciais. 

---

### **PÁGINA 36**

34 
Capítulo 1. Introdução 

cenário, justifica-se a construção de um sistema que realize a classificação do produto em conformidade com as regulamentações, mas eliminando a subjetividade e o gargalo logístico do processo.  Este trabalho busca, portanto, responder à seguinte pergunta:  Até que ponto arquiteturas de aprendizado profundo, incluindo Redes Neurais Convolucionais e Transformers, podem automatizar a classificação visual do algodão em um ambiente industrial, equilibrando acurácia e eficiência computacional? 

Assim, o objetivo geral deste trabalho é desenvolver e avaliar um sistema de visão computacional para a classificação automática do tipo de algodão, comparando o desempenho de arquiteturas CNN e Transformer.  Para atingir este objetivo, foram definidos os seguintes objetivos específicos: 

* Construir um dataset de imagens de algodão, capturadas sob iluminação controlada, representativo das classes de maior relevância comercial no cerrado brasileiro. 
* Avaliar e comparar o desempenho de diferentes arquiteturas de aprendizado profundo (ResNet, VGGNet, Inception, EfficientNet, ViT, Swin-T e ConvNeXt) em termos de acurácia, F1-Score e matriz de confusão. 
* Analisar a relação entre a complexidade computacional dos modelos (número de parâmetros e tempo de inferência) e sua eficácia na classificação, identificando a arquitetura com o melhor balanço para aplicação industrial. 

Este trabalho está organizado em X capítulos. O Capítulo 2 apresenta uma revisão da literatura sobre a classificação de algodão e as arquiteturas de visão computacional.  O Capítulo 3 detalha a metodologia... e assim por diante. 

---

### **PÁGINA 37**

35 
CAPÍTULO 2 
REVISÃO DA LITERATURA 

Este capítulo tem como objetivo consolidar a linha de raciocínio fundamental para a elaboração dessa dissertação.  Primeiramente será apresentado um panorama geral sobre o agronegócio brasileiro, com foco na cultura do algodão, abordando pontos como os principais desafios da produção, sua importância e seus destinos comerciais.  Em seguida, detalha-se o processo de avaliação da qualidade do algodão, com ênfase nos padrões internacionais, regulamentações específicas no Brasil e os processos adotados na indústria.  Na sequência, será contextualizado o tema dos sistemas de visão computacional, com aplicações no setor da agricultura, especialmente na classificação automatizada.  Por fim, o capitulo discutirá o papel da inteligência artificial e os principais modelos empregados em sistemas de visão computacional, culminando na identificação da lacuna de pesquisa que este trabalho se propõe a preencher. 

**2.1 Agronegócio Brasileiro e o Algodão** 
Esta seção apresenta a fundamentação teórica sobre a cadeia produtiva do algodão no Brasil, contextualizando o cenário onde a solução proposta nesta dissertação será aplicada.  Para o desenvolvimento de sistemas de visão computacional eficazes na classificação de fibras naturais, é imprescindível compreender a origem dos defeitos visuais e das impurezas que determinam a classificação da qualidade. 

---

### **PÁGINA 38**

36 
Capítulo 2. Revisão da Literatura 

subseções a seguir mapeiam o fluxo agroindustrial, desde a colheita mecanizada e o beneficiamento até os gargalos logísticos, identificando como cada etapa do processo produtivo influencia as propriedades físicas da pluma e impõe desafios específicos à inspeção automática de qualidade. 

**2.1.1 Agricultura no Brasil** 
A agricultura brasileira já se consolidou como uma das maiores do mundo, fornecendo produtos essenciais para diversos setores da economia global, como a alimentação, a bioenergia, o setor têxtil entre outros.  Essa consolidação é um reflexo de investimentos em inovações tecnológicas, que começaram com a terceira revolução agrícola (Revolução Verde) e se estendem até hoje com a disseminação da agricultura digital, integrando a agricultura de precisão e ferramentas como inteligência artificial para aumento da produtividade e sustentabilidade (ABBADE, 2014; BASSO; NEVES; SA, 2024).  O setor tem se destacado como um dos mais robustos, representando 23% da economia nacional (FILHO et al., 2024), com exportações atingindo recordes históricos.  Mesmo em cenários de queda nos preços das commodities e desafios climáticos, como a seca e fenômenos como o El Niño (LIN; QIAN, 2019) e gargalos logísticos (GONÇALVES et al., 2024), o Brasil manteve o crescimento no valor total exportado.  Em 2024, a expectativa é de que o país alcance US\$ 168,1 bilhões em exportações de produtos agrícolas, impulsionado por fatores como a desvalorização do real frente ao dólar, o que aumenta a competitividade dos produtos brasileiros no mercado internacional (CARDOSO et al., 2024).  No contexto da produção agrícola brasileira, a tecnologia desempenhou um papel essencial na adaptação do país a diferentes tipos de culturas, permitindo o cultivo de variedades mais resistentes e otimizadas para o clima tropical (BASSO; NEVES; SA, 2024).  Além disso, a introdução de biotecnologia e o avanço em estudos moleculares têm possibilitado o desenvolvimento de culturas com maior produtividade e resistência a pragas (DAS et al., 2023), sendo o algodão (Gossypium hirsutum) um exemplo significativo e cuja produção se beneficia de tais avanços, como resistência genética e técnicas de cultivo adaptadas ao clima e solo brasileiros (RIBEIRO et al., 2017; RIBEIRO et al., 2019; RIBEIRO et al., 2022). 

---

### **PÁGINA 39**

2.1. Agronegócio Brasileiro e o Algodão 
2.1.2 A Cadeia Produtiva do Algodão 
37 

Em 2024, o Brasil se consolidou como o maior exportador mundial de algodão em volume.  As condições climáticas favoráveis, aliadas ao aumento da demanda internacional, permitiram um crescimento expressivo das exportações brasileiras de algodão.  A produção alcançou 3,2 milhões de toneladas, com a China como principal destino, e as exportações estão projetadas para somar US\$ 5,2 bilhões até o fim do ano, um aumento de 56% em relação ao ano anterior.  Isso reflete o papel estratégico do algodão na pauta exportadora do agronegócio brasileiro, que vai além da alimentação, abrangendo setores como o têxtil e de bioenergia (CARDOSO et al., 2024).  A cadeia produtiva do algodão é caracterizada por um fluxo agroindustrial complexo, onde a valoração final do produto depende intrinsecamente da preservação das características da fibra ao longo das etapas de produção, beneficiamento e classificação.  A seguir, descrevem-se as etapas críticas que influenciam a qualidade visual e intrínseca da pluma. 

**2.1.2.1 Produção no Campo: O Impacto da Colheita na Pureza da Fibra** 
A qualidade da fibra de algodão começa no manejo da lavoura.  Fatores de estresse abiótico e o manejo da desfolha influenciam diretamente parâmetros fisiológicos da planta, determinando a maturidade e a resistência da fibra colhida (SEZENER, 2021).  No entanto, um dilema crítico para a classificação visual reside na natureza do método de colheita.  Em Mato Grosso, estado responsável por mais de 56% da produção nacional, a colheita é plenamente mecanizada.  Embora o sistema de fusos (picker) seja majoritário devido à sua seletividade superior, o sistema de arranque (stripper) ainda é utilizado em nichos de cultivo adensado visando a redução de custos operacionais.  Independentemente do maquinário, a mecanização impõe um trade-off na qualidade: FERREIRA, FIORESE e SILVA (2013) destacam que, enquanto a colheita manual apresenta perdas e impurezas em torno de 5%, a colheita mecânica eleva esse patamar para 15 a 17%.  Essa carga de material estranho impacta diretamente a análise automática.  A ação mecânica, mesmo no sistema picker, introduz contaminantes biológicos e promove a formação de neps, fenômeno exacerbado no sistema stripper, que agrega cascas, galhos, 

---

### **PÁGINA 40**

38 
Capítulo 2. Revisão da Literatura 

folhas e brácteas à pluma (KAZAMA et al., 2016; SILVA et al., 2007).  A presença destes contaminantes heterogêneos é descrita por FERREIRA, FIORESE e SILVA (2013) como responsáveis pela redução da reflectância e aumento do amarelecimento da pluma, o que define a complexidade do problema para sistemas de visão computacional.  Diferente de ambientes controlados, a classificação automática exige algoritmos robustos capazes de realizar a segmentação semântica da fibra valiosa em meio a este "ruído" vegetal e às oclusões geradas pelo processo de extração mecânica. 

**2.1.2.2 Beneficiamento (Algodoeira): O Processo de Limpeza** 
Após a colheita, o algodão em caroço é transportado para a Unidade de Beneficiamento.  Esta etapa configura-se como o elo de ligação entre a produção agrícola e a indústria têxtil, sendo fundamental para a preservação das qualidades intrínsecas da fibra.  Segundo Ferreira et al. (2022), o beneficiamento moderno no Brasil passou por profundas transformações tecnológicas, substituindo modelos obsoletos por parques fabris empresariais focados em atender a demanda internacional por qualidade e rastreabilidade.  O fluxo operacional nestas unidades visa separar a fibra da semente e remover as impurezas trazidas do campo.  Conforme detalhado tecnicamente por Silva et al. (2010) ao analisarem algodoeiras em Mato Grosso, o processo típico envolve: 

* **Pré-limpeza**: Sistemas de extração e batedores que removem impurezas grosseiras (cascas e galhos) antes que o algodão entre nos descaroçadores. 
* **Descaroçamento (Gin Stand)**: O ponto crítico do processo, onde serras mecânicas separam a fibra do caroço.  $\hat{E}$ nesta etapa que ocorre o maior estresse na fibra, podendo elevar a formação de neps em até 80%. 
* **Limpeza de Pluma (Lint Cleaning)**: Uso de equipamentos como o Constelation e fluxos de ar para expulsar a sujeira fina (poeira).  Embora aumente a limpeza visual, o excesso de processamento mecânico aqui pode romper fibras e agravar a contagem de neps. 

Apesar da modernização citada por Ferreira et al. (2022), a remoção mecânica de impurezas não ocorre sem prejuízos à integridade da pluma: o processo é eficaz 

---

### **PÁGINA 41**

2.1. Agronegócio Brasileiro e o Algodão 
39 

na limpeza, mas acaba fragmentando contaminantes maiores em partículas minúsculas (pepper trash), dificultando sua detecção posterior por métodos tradicionais (MUSTAFIC; JIANG; LI, 2016).  Para sistemas de visão computacional, isso resulta em um panorama desafiador, onde a distinção entre uma impureza vegetal fragmentada e um emaranhado de fibra exige alta precisão de detecção. 

**2.1.2.3 Classificação Comercial e Rastreabilidade** 
Após o beneficiamento e a prensagem, o algodão passa pela etapa de classificação, que transcende a simples verificação técnica para atuar como o mecanismo central de regulação comercial.  Conforme destacado por Martins (2020), a qualidade da fibra é o fator determinante para a precificação, definindo se o lote receberá ágio ou deságio sobre o preço-base de mercado (como o índice CEPEA/ESALQ), dependendo de seus atributos intrínsecos (comprimento, resistência, micronaire) e extrínsecos (impurezas e cor).  Para mitigar a subjetividade da análise visual humana, o setor consolidou o uso de instrumentos de HVI (High Volume Instrument).  Segundo o relatório da International Textile Manufacturers Federation (ITMF) (2025), a padronização via HVI foi crucial para que o Brasil deixasse de ser visto apenas como um fornecedor de "commodity" e passasse a competir em mercados "premium".  No entanto, a variabilidade no campo ainda impõe barreiras severas: Souza, Souza e Ruffato (2021) demonstram que fatores como o tempo de exposição da fibra às intempéries e o atraso na colheita degradam significativamente os parâmetros de Cor (Color Grade) e aumentam o índice de Impurezas (Trash), comprometendo a classificação final e gerando prejuízos industriais. 

Paralelamente à classificação, a rastreabilidade tornou-se um requisito não-tarifário para a exportação.  O Sistema Abrapa de Identificação (SAI) evoluiu para integrar dados de campo e indústria.  De acordo com análise técnica de Lima (2025), o uso de tecnologias como blockchain no SAI garante a imutabilidade dos dados, permitindo que cada fardo carregue uma "identidade digital" auditável desde a fazenda até a fiação.  Recentemente, essa rastreabilidade foi ampliada pelo programa "SouABR", que conecta a ponta produtiva ao varejo de moda.  Segundo estudo da FGVAgro (2022), essa iniciativa permite que o consumidor final, via QR Code, acesse o histórico completo da peça, certificando a origem sustentável e as características técnicas da fibra utilizada. 

---

### **PÁGINA 42**

40 
Capítulo 2. Revisão da Literatura 

Uma análise aprofundada sobre os parâmetros técnicos do HVI e os obstáculos computacionais da classificação visual será apresentada na Seção 2.2. 

**2.1.2.4 Armazenamento e Logística** 
A logística do algodão configura-se como o gargalo final da cadeia produtiva, especialmente na conjuntura atual onde o Brasil consolidou-se como o maior exportador mundial da fibra.  Segundo o levantamento oficial da Companhia Nacional de Abastecimento (Conab) (2025), a superação dos Estados Unidos no ranking global de exportações impõe uma pressão inédita sobre a infraestrutura nacional, exigindo eficiência para escoar um volume recorde destinado majoritariamente aos mercados asiáticos.  A primeira dificuldade crítica reside na etapa de armazenagem na origem.  Diferente de outras commodities, o fardo de algodão exige condições específicas de temperatura e umidade para evitar a degradação da fibra.  No entanto, Lourenço et al. (2020) alertam para um déficit estrutural na capacidade estática de armazenagem em Mato Grosso.  O estudo aponta que o crescimento da produção agrícola no estado foi muito superior ao investimento em armazéns, forçando o uso de pátios improvisados que expõem a pluma a riscos climáticos e dificultam a segregação dos lotes por qualidade visual e HVI. 

No transporte, a matriz brasileira permanece refém do modal rodoviário para percorrer distâncias que frequentemente superam 2.000 km entre o Cerrado e os portos.  Em uma análise recente sobre a infraestrutura do agronegócio, Péra (2025) alerta que, na contramão das melhores práticas globais, o Brasil aumentou sua dependência de caminhões para o escoamento de safra (de 45% para 54% na última década), enquanto o uso de ferrovias recuou.  Essa distorção estrutural, segundo o autor, torna o frete brasileiro significativamente mais oneroso e intensivo em carbono quando comparado aos competidores norte-americanos, que possuem uma matriz de transporte integrada e eficiente para o escoamento de suas commodities.  Para mitigar a saturação histórica do Porto de Santos, novas rotas de escoamento ganharam relevância estratégica.  O estudo técnico de Coêlho (2025), publicado pelo Banco do Nordeste, destaca a ascensão do Porto de Salvador e dos terminais do Arco Norte como alternativas viáveis.  Segundo o autor, a consolidação dessas rotas reduziu o 

---

### **PÁGINA 43**

2.2. Qualidade do algodão 
41 

tempo de trânsito para a Ásia e permitiu que a produção da Bahia e do norte de Mato Grosso escoasse com menor custo operacional.  Essa descentralização é corroborada pelos dados da Agência Nacional de Transportes Aquaviários (ANTAQ) (2026), que registram um crescimento sustentado na movimentação de cargas conteinerizadas nos portos do Norte/Nordeste.  A agência reguladora aponta que a diversificação portuária não é apenas uma questão econômica, mas uma necessidade de segurança logística para garantir o cumprimento dos contratos internacionais em prazos rígidos.  Por fim, este cenário logístico complexo impõe um entrave adicional à garantia da qualidade.  A longa exposição ao transporte rodoviário e o armazenamento em pátios abertos aumentam o risco de degradação da fibra (amarelecimento por umidade e proliferação de fungos) e de contaminações externas (ruptura de embalagens e entrada de poeira).  Nesse contexto, a classificação visual realizada na origem torna-se um "retrato estático" que pode não refletir o estado do fardo no destino final.  Isso evidencia a relevância de sistemas de classificação automática baseados em visão computacional, capazes de realizar inspeções rápidas e não destrutivas nos nós logísticos (portos e recepção de fiações), assegurando que a qualidade contratada seja efetivamente a entregue. 

**2.2 Qualidade do algodão** 
A classificação e a avaliação da qualidade do algodão são processos fundamentais para a cadeia de produção e comércio têxtil, influenciando diretamente a precificação (ágio ou deságio) e a adequação da fibra aos processos industriais de fiação.  Nesta seção, examina-se o contexto histórico, os critérios normativos e a dicotomia entre os métodos de classificação visual e instrumental, destacando as limitações que impulsionam o desenvolvimento de novas tecnologias de inspeção automática. 

**2.2.1 Contexto Histórico e Econômico** 
A prática sistemática de classificação do algodão remonta ao início do século XIX, tendo Liverpool (Inglaterra) como berço comercial.  A implementação dessa padronização foi uma resposta direta à Primeira Revolução Industrial: com a modernização das 

---

### **PÁGINA 44**

42 
Capítulo 2. Revisão da Literatura 

máquinas de fiar, observou-se que a eficiência produtiva e a qualidade do fio dependiam intrinsecamente da uniformidade da fibra, impactando a margem de lucro do setor têxtil (MORAIS et al., 2021).  Inicialmente, a percepção de qualidade variava conforme o agente (produtor ou indústria), gerando assimetria de informações.  A criação de uma terminologia padronizada e de critérios objetivos tornou-se, portanto, essencial para garantir uma base de comparação justa nas transações internacionais (EARLE; DEAN, 1914). 

Nos Estados Unidos, um marco regulatório foi a adoção do United States Cotton Futures Act pela Bolsa de Algodão de Nova York, conforme discutido por Conant (1915).  Esta legislação substituiu o sistema arbitrário de diferenciação de preços por um modelo baseado nas diferenças comerciais reais do mercado à vista (spot market).  O ato visava alinhar os contratos futuros à realidade do mercado físico, mitigando distorções que prejudicavam o hedge e a confiança dos investidores.  Além disso, introduziu a supervisão governamental na identificação dos fardos, trazendo transparência ao processo.  Segundo Earle e Dean (1914), o sistema oficial de classificação norte-americano foi consolidado no início do século XX pelo Departamento de Agricultura (USDA), resultando nos Padrões Físicos Universais.  A partir da década de 1960, o USDA impulsionou o desenvolvimento de métodos instrumentais, culminando na adoção massiva do sistema HVI (High Volume Instrument).  Atualmente, embora a classificação instrumental seja predominante na produção americana, a inspeção visual manual ainda é utilizada para identificação de contaminantes específicos, sendo objeto de pesquisas contínuas para a automação total do processo (KNOWLTON, 2005; INCORPORATED, 2020). 

**2.2.2 Métodos e Padrões de Classificação no Brasil** 
No Brasil, a classificação da pluma é regida pela Instrução Normativa n.º 24/2016 do Ministério da Agricultura, Pecuária e Abastecimento (MAPA).  O regulamento harmoniza os procedimentos nacionais aos Padrões Físicos Universais, avaliando atributos como cor, presença de folhas, impurezas, contaminação de matérias estranhas e o modo de preparação (beneficiamento) (MAPA, 2016).  O processo inicia-se com uma amostragem rigorosa para garantir a representatividade do lote.  Conforme a normativa, devem ser extraídas duas amostras de lados 

---

### **PÁGINA 45**

2.2. Qualidade do algodão 
43 

opostos de cada fardo (mínimo de 150g cada). Estas são divididas longitudinalmente e combinadas para formar as amostras de trabalho: uma destinada à classificação visual e outra à instrumental.  As dimensões (25 a 30 cm de comprimento) e o acondicionamento devem assegurar que a estrutura da pluma não seja descaracterizada antes da análise. 

**2.2.2.1 A Classificação Visual e a Subjetividade Humana** 
A classificação visual é realizada por técnicos habilitados (classificadores), que utilizam como referência os Padrões Físicos Universais (Universal Standards).  Estes padrões, estabelecidos pelo USDA e adotados no Brasil, consistem em caixas contendo amostras físicas de algodão que representam as classes de cor e graus de folha (Leaf Grade) para as variedades "Upland" e "Pima", conforme ilustrado na Figura 1. 

**Figura 1** - Exemplo do Padrão Físico Universal (51.5: Cor Abaixo da Média, Branco / Grau de Folhas 5) 
Fonte: Do Autor 

O protocolo de análise é rigoroso e inicia-se pela "calibração visual" do técnico, que deve memorizar os padrões físicos antes de iniciar a classificação, podendo consultá- 

---

### **PÁGINA 46**

44 
Capítulo 2. Revisão da Literatura 

los sempre que necessário para evitar desvios de julgamento. O manuseio da amostra exige técnica específica: ela deve ser subdividida longitudinalmente em camadas, com cuidado para não descaracterizar sua estrutura.  As duas faces consideradas "piores" - ou seja, aquelas com maior incidência de defeitos visíveis são selecionadas e voltadas para cima para a avaliação final.  Neste exame, o classificador julga simultaneamente múltiplos atributos qualitativos, conforme detalhado por Abbade (2014): 

* **Cor**: Avaliada pela combinação de matiz, saturação e brilho, sendo categorizada em classes (Branco, Ligeiramente Manchado, Manchado, Tingido e Amarelo); 
* **Impurezas (Leaf Grade)**: Quantificação visual de cascas, folhas, caules e outras matérias estranhas.  O grau é atribuído em uma escala de 1 a 8, onde 1 representa a amostra mais limpa e 8 a mais suja; 
* **Preparação**: Identificação de defeitos oriundos do processo de beneficiamento, como a aspereza e a formação de neps (emaranhados de fibras); 
* **Contaminações**: Presença de materiais não vegetais (plásticos, óleos) que desclassificam o lote. 

A determinação final do "Tipo" é uma composição entre o padrão de cor e o grau de folha, registrada no Laudo de Classificação (Tabela 1). 

Apesar da padronização dos critérios, a classificação visual possui limitações inerentes à percepção humana.  Matusiak e Walawska (2010) argumentam que a análise sensorial é fortemente influenciada por fatores como cansaço visual, experiência do técnico e condições de iluminação.  Essa subjetividade gera variabilidade nos resultados, tanto entre diferentes avaliadores quanto nas decisões do mesmo técnico em momentos distintos.  Além disso, a dificuldade natural do olho humano em distinguir nuances sutis de cor ou quantificar impurezas minúsculas (pepper trash) reforça a necessidade de sistemas de Visão Computacional, que oferecem a reprodutibilidade e a precisão objetiva exigidas pela indústria têxtil. 

---

### **PÁGINA 47**

**Tabela 1 - Códigos dos tipos de Cor, Classes de Cor ou Grau de Cor do Algodão Upland** 

| | Branco | Ligeiramente Creme | Creme | Avermelhado | Amarelado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Cor Boa Média - GM (Good Middling) | 11 | 12 | 13 | | |
| Cor Estritamente Média-SM (Strict Middling) | 21 | 22 | 23 | 24 | 25 |
| Cor Média-M (Middling) | 31 | 32 | 33 | 34 | 35 |
| Cor Estritamente Abaixo da Média-SLM (Strict Low Middling) | 41 | 42 | 43 | 44 | |
| Cor Abaixo da Média-LM (Low Middling) | 51 | 52 | 53 | 54 | |
| Cor Estritamente Boa Comum-SGO (Strict Good Ordinary) | 61 | 62 | 63 | | |
| Cor Boa Comum-GO (Good Ordinary) | 71 | - | | | |
| Abaixo de Padrão | 81 | 82 | 83 | 84 | 85 |



---

### **PÁGINA 48**

46 
Capítulo 2. Revisão da Literatura 

**2.2.2.2 A Classificação Instrumental (HVI) e suas Limitações** 
Paralelamente à análise visual, a segunda amostra de trabalho é destinada à classificação instrumental.  Diferente da inspeção humana, esta etapa exige rigoroso controle ambiental: as amostras devem permanecer em repouso em bandejas teladas por, no mínimo, 24 horas em laboratório climatizado (temperatura e umidade controladas) para atingir o equilíbrio higroscópico, garantindo a reprodutibilidade das leituras físicas.  A análise é realizada em equipamentos de alto volume conhecidos como HVI (High Volume Instrument), calibrados diariamente com padrões internacionais do USDA.  A Figura 2 apresenta o modelo Uster HVI 1000, amplamente adotado pela indústria têxtil. 

**Figura 2** - Equipamento Uster HVI 1000 
Fonte: (Uster Technologies, 2024) 

O protocolo operacional do HVI automatiza a mensuração de múltiplas propriedades físicas.  O teste de Micronaire é realizado em um único corpo de prova (compressão de massa de fibras), enquanto os demais parâmetros de comprimento e resistência são obtidos através de pentes óticos e garras mecânicas em dois corpos de prova por amostra, gerando uma média estatística do lote.  Os principais parâmetros fornecidos no laudo técnico são: 

---

### **PÁGINA 49**

2.2. Qualidade do algodão 
47 

* **Comprimento da Fibra (UHML)**: Comprimento médio da metade superior das fibras. 
* **Índice de Uniformidade do Comprimento da Fibra (%UI)**: Relação entre o comprimento médio total das fibras e o comprimento médio das fibras mais longas. 
* **Resistência ou Tenacidade da Fibra (Str)**: Medida da força necessária para romper as fibras, expressa em $gf/tex$. 
* **Alongamento à Rotura da Fibra (% Elg)**: Percentual de extensão que a fibra suporta antes de romper. 
* **Micronaire da Fibra (Mic)**: Combinação de finura e maturidade da fibra. 
* **Grau de Reflectância (%Rd)**: Nível de luminosidade e cor branca refletida pelas fibras. 
* **Grau de Amarelamento (+b)**: Índice que expressa o nível de amarelamento da fibra. 
* **Índice de Fibras Curtas (%SFI)**: Percentual de fibras com comprimento inferior a 0,50 polegadas. 
* **Índice de Consistência da Fiação (SCI)**: Valor calculado com base nas propriedades físicas das fibras e sua correlação com a performance de fiação. 

Embora o HVI tenha trazido objetividade às transações comerciais, ele apresenta limitações críticas sob a ótica da classificação final.  Além do alto custo de aquisição e manutenção (BRAZIL, 2021), o equipamento fornece métricas baseadas em médias globais da amostra.  Isso significa que, embora o HVI quantifique com precisão a área total de impurezas, ele não captura a percepção visual holística, como a distribuição espacial da sujeira, a textura e o contraste com a fibra, utilizada pelos classificadores humanos para determinar o "Tipo" comercial.  Essa lacuna de percepção é justamente o nicho onde sistemas de Visão Computacional baseados em Aprendizado Profundo se destacam, pois são capazes de correlacionar padrões visuais complexos diretamente com os graus de qualidade padronizados, oferecendo uma alternativa automatizada e consistente à subjetividade humana. 

---

### **PÁGINA 50**

48 
**2.2.2.3 A Relação entre o Visual e o Instrumental** 
Capítulo 2. Revisão da Literatura 

A integração entre a subjetividade da análise visual e a objetividade instrumental baseia-se historicamente nos estudos de Nickerson e Newton (1958).  Buscando traduzir a percepção humana para grandezas físicas mensuráveis, os autores desenvolveram o colorímetro Nickerson-Hunter, estabelecendo as bases para o atual diagrama de cores utilizado pelo sistema HVI.  Conforme ilustrado na Figura 3, a relação é estabelecida em um plano bidimensional onde: 

* O eixo vertical (%Rd) representa a Reflectância (brilho), correlacionando-se com a escala de cinzas (do branco ao cinza escuro); 
* O eixo horizontal (+b) representa o Grau de Amarelamento, indicando a saturação da cor amarela na fibra. 

As interseções destes valores formam clusters que correspondem às classes visuais tradicionais (ex: Middling, Strict Low Middling).  Dessa forma, o HVI consegue converter leituras sensoriais em uma "nota" comercial compatível com o julgamento humano. 

---

### **PÁGINA 51**

49 
**Figura 3** - Diagrama de Cores relacionando parâmetros físicos (%Rd e +b) com a classificação visual. 

(Eixos e rótulos detalhados: +b do amarelamento de 4 a 18; RD % Reflectance de 40 a 90; Classes: WHITE, LT SP, SPOTTED, TINGED, Y.S.; Códigos numéricos de clusters internos do USDA.) 

Fonte: Cotton Incorporated (2024) 

Entretanto, essa correlação robusta observada na análise de cor não se verifica com a mesma eficácia para os parâmetros de Impurezas e Preparação.  Enquanto o HVI mensura a sujeira por contagem de partículas e área ocupada, o classificador humano avalia a aparência geral, considerando o contraste, o tamanho e a natureza do resíduo 

---

### **PÁGINA 52**

50 
Capítulo 2. Revisão da Literatura 

(folha, casca ou caule). Estudos indicam que a correlação entre o "Grau de Folha" visual e a leitura de "%Area" do HVI pode ser baixa em lotes com sujeira heterogênea.  Essa discrepância evidencia uma lacuna tecnológica: o instrumento é preciso na física (cor), mas limitado na interpretação de padrões complexos (sujeira e textura).  $\hat{E}$ neste cenário que se insere a proposta desta dissertação: utilizar técnicas avançadas de Visão Computacional para mimetizar a percepção cognitiva humana, buscando uma classificação de qualidade que considere não apenas a estatística da sujeira, mas sua morfologia e distribuição visual na pluma. 

**2.3 Visão Computacional Aplicada à Agricultura** 
Nesta seção será explorada com mais profundidade a disciplina de visão computacional, os sistemas de visão computacional e seus conceitos fundamentais, o processamento de imagens e aplicação da visão computacional no agro, com ênfase nas tecnologias e métodos aplicados à cultura do algodão.  A Visão Computacional, embora frequentemente associada às inovações recentes da Inteligência Artificial, possui raízes que remontam a meados do século XX.  Segundo a revisão histórica de Oliveira e Honorato (2024), a área foi idealizada inicialmente em 1955, sob a premissa de equipar computadores com capacidades sensoriais análogas aos "olhos e ouvidos" humanos.  Nas décadas seguintes, especialmente nos anos 1970, pesquisadores acreditavam que a emulação da visão biológica seria um marco rapidamente alcançável.  Neste contexto, destaca-se o trabalho seminal de Marr (1976) no campo da neurofisiologia, que propôs uma abordagem computacional para a visão, formulando algoritmos para a interpretação de cenas que serviram como alicerce para grande parte da pesquisa subsequente (VAINA, 2004).  Contudo, a evolução da área revelou que o processamento visual era intrinsecamente mais complexo do que o previsto, dada a lacuna de conhecimento sobre como o próprio cérebro humano interpreta estímulos visuais.  Após décadas de avanços progressivos e o aumento do poder computacional, a Visão Computacional consolidou-se como a disciplina científica dedicada ao desenvolvimento de métodos que permitem às máquinas não apenas capturar, mas interpretar e "compreender" imagens com um nível de abstração semelhante à percepção humana (BALLARD; BROWN, 1982). 

---

### **PÁGINA 53**

2.3. Visão Computacional Aplicada à Agricultura 
**2.3.1 Fundamentos de um Sistema de Visão Computacional** 
51 

Um sistema de visão computacional é a integração de hardware e software projetada para simular a percepção visual humana.  Saldaña et al. (2013) definem que tais sistemas visam extrair e analisar informações de objetos de forma automática, objetiva e não invasiva.  A arquitetura típica compõe-se de três pilares fundamentais: o dispositivo de aquisição de imagens (sensores), o sistema de iluminação e a unidade de processamento (software). 

**2.3.1.1 Captura de imagens** 
A aquisição da imagem é o primeiro estágio do processo, onde a radiação eletromagnética refletida pelo objeto é convertida em dados digitais.  Embora existam sensores baseados em ultrassom, raios-X e espectroscopia NIR, as câmeras digitais operando no espectro visível são os dispositivos predominantes na agricultura de precisão (BROSNAN; SUN, 2004).  No núcleo destas câmeras, a fototransdução é realizada por dois tipos principais de sensores: CCD (Charge-Coupled Device) e CMOS (Complementary Metal-Oxide-Semiconductor).  As câmeras CCD são historicamente reconhecidas pela alta fidelidade de imagem e sensibilidade em condições de baixa luminosidade.  Isso decorre de sua arquitetura, que privilegia uma maior área fotossensível (fator de preenchimento) e uniformidade na resposta dos pixels, resultando em baixo ruído e ampla faixa dinâmica.  Tais características as tornaram padrão em aplicações científicas de alta precisão.  Contudo, seu alto consumo energético e a complexidade dos circuitos externos limitam sua portabilidade (LITTWILLER, 2001).  Em contrapartida, a tecnologia CMOS evoluiu drasticamente nas últimas décadas.  Sua arquitetura permite a integração de conversores e amplificadores diretamente no pixel, reduzindo o consumo de energia, o tamanho físico e o custo de fabricação.  Embora versões antigas apresentassem maior ruído ("ruído de padrão fixo"), sensores CMOS modernos atingiram níveis de qualidade comparáveis aos CCDs, tornando-se a escolha dominante para dispositivos móveis e sistemas embarcados no campo (BIGAS et al., 2006).  A Tabela 2 resume as diferenças técnicas entre as tecnologias. 

---

### **PÁGINA 54**

52 
Capítulo 2. Revisão da Literatura 

**Tabela 2 - Resumo comparativos das principais características entre sensores CCD e CMOS** 

| Característica | CCD | CMOS |
| :--- | :--- | :--- |
| Consumo de energia | Alto | Baixo |
| Custo | Alto | Baixo |
| Integração em chip | Não | Sim |
| Flexibilidade | Limitada | Alta |
| Faixa dinâmica | Alta | Limitada |
| Sensibilidade | Alta | Limitada |
| Ruído | Baixo | Alto |
| Qualidade de imagem | Alta | Limitada |
| Velocidade | Limitada | Alta |
| Tamanho físico do sistema | Maior | Menor |
| Aplicações | Alta qualidade, ciências | Dispositivos móveis, segurança |


Fonte: Bigas et al. (2006) 

No entanto, para a aplicação específica da classificação de algodão, a escolha do sensor impõe desafios críticos que vão além da arquitetura de fabricação.  A natureza física da pluma exige características sensoriais específicas para garantir a correlação com os padrões comerciais: 

1.  **Faixa Dinâmica (Dynamic Range)**: O algodão beneficiado possui alta refletância (brilho), criando um cenário de alto contraste onde as impurezas são significativamente mais escuras que a fibra.  Li et al. (2025) destacam que a captura de imagens neste cenário enfrenta o desafio de sombras e brilho excessivo;  sensores com baixa faixa dinâmica tendem a saturar nos brancos ou perder detalhes nas áreas sombreadas das impurezas, exigindo algoritmos robustos de segmentação (como o Color-Unet) para garantir a consistência da medição de cor. 
2.  **Resolução Espacial**: A detecção de micro-impurezas (pepper trash) impõe um desafio de amostragem. Li et al. (2025) argumentam que métodos tradicionais falham em detectar esses pequenos objetos devido à complexidade do fundo e à mistura de pixels.  O uso de sensores de alta resolução, aliados a redes neurais profundas, torna-se mandatório para distinguir contaminantes minúsculos que, em resoluções menores, seriam invisíveis ou confundidos com ruído. 

---

### **PÁGINA 55**

2.3. Visão Computacional Aplicada à Agricultura 
53 

3.  **Fidelidade de Cor**: A classificação comercial depende estritamente dos parâmetros de Amarelamento (+b) e Reflectância (Rd).  Segundo Li et al. (2023), a simples captura RGB não garante a correlação com o padrão HVI.  É necessária uma calibração rigorosa e a conversão para espaços de cor perceptuais (como CIE $L^{*}a^{*}b^{*}$ ou Hunter) para que o sistema de visão atinja coeficientes de determinação superiores a 0,88 quando comparado aos colorímetros industriais, evitando erros de graduação que resultam em prejuízo financeiro. 

Além do sensor, a integridade da análise depende do formato de armazenamento.  Em aplicações científicas, formatos de compressão sem perdas (lossless), como PNG, BMP ou TIFF, são preferíveis, pois preservam o valor exato de cada pixel.  Formatos com perdas (lossy), como o popular JPEG, utilizam algoritmos que descartam informações de alta frequência para reduzir o tamanho do arquivo.  Para a classificação de algodão, onde a textura fina e a presença de micro-impurezas são críticas, os artefatos de bloco gerados pela compressão JPEG podem ser interpretados erroneamente pelos algoritmos como ruído ou defeitos, comprometendo a precisão do modelo (WALT et al., 2014). 

**2.3.1.2 Iluminação** 
A iluminação desempenha um papel essencial nos sistemas de visão computacional, pois define as propriedades ópticas dos objetos a serem analisados.  A luz incidente pode ser absorvida, refletida ou transmitida, influenciando diretamente a qualidade das imagens capturadas.  Para otimizar o desempenho do sistema de visão, a escolha e a configuração do sistema de iluminação devem ser adaptadas à aplicação específica.  Essa seleção cuidadosa reduz a necessidade de etapas complexas de processamento de imagem, aumentando a eficiência e a precisão das análises (CHEN; CHAO; KIM, 2002).  Zhang e Li (2014) destacam a importância de fontes de luz que realcem as características dos objetos e permitam distinções claras entre as partes detectadas e outras áreas.  A eficiência e a precisão de um sistema de visão computacional dependem significativamente da qualidade da iluminação, que deve fornecer luz uniforme e estável, além de evitar reflexos e sombras que possam comprometer a análise.  No contexto específico da classificação de algodão, a iluminação desempenha um papel duplo e, por vezes, antagônico, exigindo um compromisso técnico (trade-off): 

---

### **PÁGINA 56**

54 
Capítulo 2. Revisão da Literatura 

* **Para Análise de Cor (Simulação de HVI)**: A classificação comercial baseia-se na reflectância (%Rd) e no amarelamento (+b).  Rady et al. (2025), ao desenvolverem sistemas de visão para graduação de algodão egípcio, demonstraram que a consistência na classificação de cor exige uma iluminação difusa e homogênea.  O uso de luz direcional cria sombras e reflexos especulares ("brilhos") na fibra que alteram os valores RGB capturados, impedindo a correlação direta com os padrões físicos universais. 
* **Para Detecção de Impurezas e Robustez**: Paradoxalmente, a luz difusa que favorece a cor tende a "suavizar" a textura, ocultando defeitos tridimensionais.  Verma et al. (2026) destacam que a variabilidade de iluminação é um dos principais desafios para a segmentação robusta de capulhos e impurezas em cenários complexos.  Para detectar contaminantes que possuem a mesma cor da fibra (como plásticos claros) ou defeitos de preparação (neps), uma componente de luz direcional ou rasante é fundamental para gerar micro-sombras que revelam a morfologia do objeto, permitindo que modelos de detecção superem as limitações da simples análise de cor. 

Atualmente, a tecnologia LED consolidou-se como padrão para solucionar esse dilema.  Sua estabilidade espectral e capacidade de controle rápido permitem o desenvolvimento de sistemas de iluminação híbridos ou multiespectrais, que alternam entre configurações difusas e direcionais para capturar o máximo de informação da amostra em uma única inspeção (CUBERO et al., 2010).  Dessa forma, o projeto do sistema de iluminação para esta dissertação não buscou apenas a visibilidade, mas a fidelidade fotométrica, garantindo que a imagem digital servisse como um proxy confiável tanto para a avaliação cromática quanto para a inspeção morfológica da qualidade. 

**2.3.1.3 Software** 
O componente de software atua como o cérebro do sistema de visão computacional, responsável não apenas por orquestrar o hardware de captura e iluminação, mas principalmente por implementar os algoritmos de interpretação visual.  Historicamente, o desenvolvimento baseava-se em bibliotecas de processamento de imagem clássico (ou 

---

### **PÁGINA 57**

2.3. Visão Computacional Aplicada à Agricultura 
55 

rule-based), como o OpenCV (Open Source Computer Vision Library). Estas ferramentas exigiam que o engenheiro definisse manualmente os descritores matemáticos para extrair bordas, cores ou formas geométricas (BRADSKI, 2000).  No entanto, o estado da arte na agricultura de precisão sofreu uma mudança de paradigma com a consolidação do Aprendizado Profundo (Deep Learning).  Conforme revisado por Mustofa et al. (2023), o foco do desenvolvimento de software migrou da engenharia de características manual para a curadoria de dados e o treinamento de Redes Neurais Convolucionais (CNNs).  Neste novo cenário, o software não segue mais regras estáticas ("se o pixel é verde"), mas aprende padrões complexos a partir de exemplos, exigindo frameworks robustos para o treinamento e inferência dos modelos.  Atualmente, dois ecossistemas dominam o desenvolvimento de software para visão agrícola: 

* **Frameworks de Treinamento (PyTorch/TensorFlow)**: São as plataformas onde os modelos são construídos. Uma revisão recente de Zhang et al. (2025) destaca que, embora o TensorFlow tenha sido pioneiro na produção industrial, o PyTorch ganhou predominância na pesquisa agrícola devido à sua flexibilidade e facilidade de depuração (debugging), permitindo a prototipagem rápida de novas arquiteturas para detecção de doenças e classificação de qualidade. 
* **Arquiteturas de Detecção (YOLOv8/Transformers)**: O software moderno implementa arquiteturas pré-treinadas que oferecem o equilíbrio ideal entre velocidade e precisão.  Jelali (2025) destacam a arquitetura YOLOv8 (You Only Look Once, versão 8) como o padrão atual para monitoramento agrícola em tempo real, superando antecessores na detecção de pequenos objetos em cenários não estruturados uma característica vital para identificar impurezas minúsculas no algodão. 

Por fim, uma camada crítica do software frequentemente negligenciada é a ferramenta de anotação e gestão de dados.  Como a performance da IA depende estritamente da qualidade dos exemplos fornecidos ("Garbage In, Garbage Out"), o uso de plataformas para rotulagem consistente de imagens tornou-se um pré-requisito.  Yang et al. (2025) argumentam que a falta de anotações precisas é hoje o maior gargalo para a aplicação de visão computacional no campo, exigindo softwares que facilitem a marcação semân- 

---

### **PÁGINA 58**

56 
Capítulo 2. Revisão da Literatura 

tica de defeitos e o aumento de dados para robustecer o modelo contra variações de iluminação. 

**2.3.2 Processamento e Análise de Imagens: Do Clássico ao Aprendizado Profundo** 
O processamento e a análise de imagens constituem o núcleo algorítmico dos sistemas de visão, transformando a matriz numérica bruta capturada pelo sensor em informações semânticas para a tomada de decisão.  A literatura clássica, consolidada por Gonzalez e Woods (2017), organiza esse fluxo em uma hierarquia de três níveis: baixo, médio e alto nível.  Embora a ascensão do Aprendizado Profundo (Deep Learning) tenha automatizado a transição entre essas etapas, a compreensão dessa estrutura permanece fundamental para o design de sistemas robustos. 

**2.3.2.1 Processamento de Baixo Nível: Pré-processamento** 
O processamento de baixo nível engloba operações onde tanto a entrada quanto a saída são imagens.  O objetivo primordial é o melhoramento da qualidade visual e a restauração de dados para facilitar as etapas subsequentes.  Técnicas comuns incluem a correção radiométrica (uniformização da iluminação), a redução de ruído via filtros espaciais e a correção geométrica de distorções da lente.  No contexto moderno de redes neurais, esta etapa evoluiu para incluir o aumento de dados (Data Augmentation) e a normalização estatística.  Shorten e Khoshgoftaar (2019) destacam que, para aplicações agrícolas onde a variabilidade de campo é alta, aplicar transformações geométricas e de cor nas imagens de treinamento é crucial para evitar o superajuste (overfitting) e garantir que o modelo generalize bem para diferentes condições de iluminação. 

**2.3.2.2 Processamento de Nível Intermediário: Segmentação e Representação** 
Neste estágio, a operação transita de imagens para atributos. A tarefa central é a segmentação, que visa particionar a imagem em regiões de interesse: No caso do algodão, separar a fibra do fundo e isolar as impurezas. 

---

### **PÁGINA 59**

2.3. Visão Computacional Aplicada à Agricultura 
57 

Abordagens tradicionais baseavam-se em limiarização (thresholding) e detecção de bordas (SONKA; HLAVAC; BOYLE, 1993). Contudo, Li et al. (2025) argumentam que métodos clássicos falham na classificação de algodão devido à complexidade textural: sombras na fibra podem ser confundidas com sujeira, e impurezas claras (como plásticos) não possuem contraste suficiente para uma limiarização simples.  Essa limitação impulsionou a adoção de Redes Neurais Convolucionais (CNNs).  Diferente da extração manual de atributos (forma, textura, momentos), as CNNs aprendem hierarquicamente as representações necessárias: as primeiras camadas da rede atuam como filtros de borda e textura (nível intermediário), enquanto as camadas profundas agem como segmentadores semânticos, identificando padrões complexos que escapam à modelagem matemática tradicional (LECUN; BENGIO; HINTON, 2015). 

**2.3.2.3 Processamento de Alto Nível: Interpretação e Classificação** 
O nível final envolve a "compreensão" da cena, atribuindo significado aos objetos segmentados.  Na classificação de qualidade do algodão, isso não significa apenas detectar uma partícula, mas interpretá-la semanticamente: classificar se é uma folha, um caule ou um defeito de preparação (nep) e, com base no conjunto, atribuir uma nota global ao fardo (Tipo/Grau).  Enquanto sistemas antigos utilizavam classificadores estatísticos (como SVM ou KNN) sobre vetores de características fixos, os sistemas modernos realizam a classificação de forma integrada.  O modelo aprende a correlacionar a distribuição espacial e morfológica das impurezas diretamente com os padrões de qualidade comercial, emulando o processo cognitivo de um classificador humano experiente (PACAL; AL., 2024). 

**2.3.3 Estado da Arte: Da Inspeção de Campo à Classificação de Fibras** 
A aplicação da visão computacional no agronegócio ultrapassou a fase experimental para se tornar uma ferramenta essencial na garantia de qualidade e produtividade.  A literatura recente evidencia uma migração das técnicas clássicas de processamento de imagem para abordagens baseadas em Aprendizado Profundo (Deep Learning), capazes 

---

### **PÁGINA 60**

58 
Capítulo 2. Revisão da Literatura 

de lidar com a variabilidade natural dos produtos agrícolas (PATRÍCIO; RIEDER, 2018; TIAN et al., 2020).  Abaixo, discute-se como essa tecnologia evoluiu do monitoramento de campo para a rigorosa inspeção de qualidade industrial, nicho onde este trabalho se insere. 

**2.3.3.1 Monitoramento e Detecção de Padrões em Campo** 
No ambiente de produção, a visão computacional é amplamente utilizada para a detecção de anomalias biológicas. Ahad et al. (2023) e Ritharson et al. (2023) demonstraram que Redes Neurais Convolucionais (CNNs) podem superar a acuidade humana na identificação de doenças, atingindo acurácias superiores a 98% em culturas de arroz.  A tendência atual é a implementação desses modelos em arquiteturas leves (como YOLOv8) para processamento em tempo real (SHWETHA; BHAGWAT; LAXMI, 2024).  Essa capacidade de detecção estende-se à cultura do algodão ainda no campo. Singh et al. (2023), por exemplo, desenvolveram algoritmos para identificar capulhos maduros em tempo real, visando a automação da colheita.  No entanto, embora a detecção em campo esteja avançada, o maior desafio tecnológico reside na etapa seguinte: a avaliação da qualidade do produto colhido. 

**2.3.3.2 Inspeção de Qualidade e Padronização Pós-Colheita** 
A avaliação de qualidade de produtos agrícolas, historicamente subjetiva, tem sido transformada pela automação visual.  Desde os trabalhos pioneiros de Majumdar e Jayas (1999) na classificação de cereais por textura, a tecnologia evoluiu para detectar defeitos sutis imperceptíveis em esteiras de alta velocidade.  Zhu et al. (2024) aplicaram técnicas de luz estruturada para identificar danos mecânicos em maçãs com alto índice de intersecção $(IoU>91\%)$, validando o uso de visão computacional para a tipificação comercial.  No contexto específico da fibra de algodão, a pureza e a cor são os atributos mais críticos.  A detecção de contaminantes (foreign matter) é um problema clássico revisitado por Wang et al. (2023), que implementaram a arquitetura YOLOv5 para distinguir fibras de materiais estranhos (cascas, plásticos) com alta velocidade.  A complexidade desta tarefa assemelha-se à detecção de ervas daninhas em campo, onde a distinção entre o 

---

### **PÁGINA 61**

2.4. Classificação Automatizada do Algodão com Visão Computacional 
59 

"alvo" e o "fundo" exige robustez contra variações de iluminação e oclusão (JUWONO et al., 2023; SUBEESH et al., 2022).  Apesar desses avanços pontuais na detecção de contaminantes, ainda existe uma lacuna na correlação direta entre os padrões visuais aprendidos por máquinas e as classes comerciais manuais (Tipo e Cor) utilizadas na comercialização da pluma, conforme estabelecido pelos padrões universais.  $\hat{E}$ nesta intersecção entre a avançada capacidade de detecção de objetos (YOLO/CNNs) e a necessidade de padronização comercial que esta dissertação atua. 

**2.4 Classificação Automatizada do Algodão com Visão Computacional** 
A classificação da pluma de algodão, historicamente dependente da inspeção manual ou de instrumentos laboratoriais como o HVI (High Volume Instrument), atravessa uma profunda mudança de paradigma impulsionada pela Inteligência Artificial.  A convergência entre processamento de imagem, aprendizado de máquina e tecnologias conectadas (IoT/Edge Computing) tem se consolidado como uma solução transformadora para superar as limitações dos métodos tradicionais, que frequentemente exigem mão de obra intensiva, consomem muito tempo e estão sujeitos à subjetividade.  A literatura recente evidencia que a aplicação dessas tecnologias evoluiu substancialmente, expandindo-se do monitoramento agrícola em campo para a automação rigorosa do "processamento inteligente" e da inspeção de qualidade na indústria.  O uso do Aprendizado Profundo (Deep Learning) assumiu um papel dominante nesse cenário, viabilizando a extração de características complexas da fibra que abordagens matemáticas ou estatísticas convencionais não conseguiam modelar com precisão.  Nesta seção, explora-se a trajetória tecnológica dessa automação, abordando desde as limitações físicas dos sensores de captura até a evolução dos modelos preditivos.  A discussão abrange a transição do aprendizado de máquina clássico para o atual estado da arte baseado em redes neurais convolucionais, detalhando como a pesquisa moderna tem solucionado a avaliação automatizada dos três pilares da qualidade comercial do fardo: cor, impurezas (trash) e preparação. 

---

### **PÁGINA 62**

60 
Capítulo 2. Revisão da Literatura 

**2.4.1 Limitações Sensoriais e Multiespectrais** 
Embora câmeras RGB sejam predominantes devido ao baixo custo, a literatura aponta limitações físicas críticas na detecção de certos contaminantes.  Mustafic, Jiang e Li (2016) demonstraram que materiais como plásticos transparentes e papéis brancos possuem assinaturas visuais quase idênticas à fibra de algodão no espectro visível, tornando-os virtualmente invisíveis para sensores convencionais.  A solução proposta pelos autores envolve o uso de imagem hiperespectral de fluorescência (UV), capaz de distinguir a composição química de contaminantes não-botânicos com precisão superior a 90%.  Paralelamente, Li et al. (2024) exploraram o uso de Espectroscopia de Infravermelho Próximo (NIR) combinada com redes neurais ("Cotton-Net") para estimar o teor de impurezas.  Embora o método seja eficaz para determinar a composição química global da amostra, ele carece de informação espacial detalhada para classificar a morfologia dos defeitos, reafirmando a necessidade da visão computacional para a avaliação física da "aparência" do fardo. 

**2.4.2 O Aprendizado de Máquina e a Variabilidade Intra-Amostra** 
A transição da estatística clássica para o Aprendizado de Máquina (Machine Learning) marcou a primeira etapa da automação robusta.  Fisher et al. (2023a) investigaram a classificação de algodão egípcio utilizando algoritmos de Random Forest, superando abordagens baseadas em SVM e Redes Neurais Artificiais (ANN) clássicas.  Uma contribuição fundamental deste estudo foi a introdução da "variância intra-amostra" como preditor de qualidade: os autores provaram que a uniformidade da cor ao longo da superfície da amostra é tão determinante para a classificação comercial quanto a média dos valores de reflectância. 

Contudo, a eficácia desses modelos esbarra em um gargalo humano: a qualidade da rotulagem dos dados (ground truth).  Fisher et al. (2023b) alertam que a subjetividade e a fadiga dos classificadores humanos introduzem ruído nos dados de treinamento.  Para mitigar o alto custo e a inconsistência da anotação manual, os autores propuseram o uso de **Aprendizado Ativo** (Active Learning), onde o sistema solicita intervenção humana apenas para imagens de alta incerteza, reduzindo drasticamente o volume de 

---

### **PÁGINA 63**

2.4. Classificação Automatizada do Algodão com Visão Computacional 
61 

dados necessários enquanto mantém a acurácia do modelo. 

**2.4.3 O Estado da Arte: Deep Learning e Segmentação Semântica** 
Atualmente, a fronteira do conhecimento reside na aplicação de Redes Neurais Convolucionais (CNNs) para tarefas de segmentação semântica e detecção de objetos, focando na correlação direta com os parâmetros do HVI. 

No que tange à Análise de Cor, o principal desafio é a interferência de sombras e sujeira na leitura dos pixels.  Li et al. (2025) propuseram o método inovador "Color-Unet". Diferente das abordagens que calculam a média de toda a imagem, esta rede realiza a segmentação semântica para isolar a fibra pura, excluindo pixels de impurezas e áreas sombreadas.  A medição de cor realizada apenas na "máscara de fibra limpa" resultou em uma correlação significativamente maior com os valores de referência do HVI. 

Para a detecção de Impurezas (Trash), a arquitetura YOLO (You Only Look Once) estabeleceu-se como padrão. Li et al. (2025) desenvolveram o modelo "Cotton-YOLO", uma versão otimizada com módulos de atenção (CBAM) projetada especificamente para detectar micro-impurezas (pepper trash) que redes genéricas falham em identificar.  Avançando para a quantificação precisa, Jiang et al. (2024) introduziram o "Cotton-YOLO-Seg", que não apenas detecta o objeto, mas realiza a segmentação de instância (pixel a pixel).  Isso permite o cálculo exato da área ocupada por cada partícula, viabilizando a automação do parâmetro "% Area" do HVI com precisão inédita. 

Por fim, a avaliação da Preparação (irregularidades como neps) também se beneficia dessas arquiteturas. Arachchi et al. (2024) aplicaram CNNs baseadas em transfer learning (como Inception V3) para classificar defeitos em fios de algodão, demonstrando que padrões morfológicos sutis de emaranhamento podem ser aprendidos eficazmente por redes profundas, uma lógica que se estende à avaliação da pluma crua. 

---

### **PÁGINA 64**

62 
Capítulo 2. Revisão da Literatura 

**2.5 Arquiteturas de Aprendizado Profundo para Classificação de Imagens** 
A eficácia dos modernos sistemas de visão computacional não reside apenas no volume de dados ou no poder de processamento bruto, mas fundamentalmente no design das arquiteturas de extração de características.  Na classificação da pluma de algodão, o algoritmo enfrenta um desafio multiescala crônico: ele precisa possuir um campo receptivo global para avaliar a uniformidade da cor e a distribuição das fibras, ao mesmo tempo em que necessita de altíssima sensibilidade local para detectar micro-impurezas (pepper trash) e defeitos de preparação (neps).  Historicamente, a superação desses desafios acompanhou a própria evolução das topologias de redes neurais.  A transição de redes rasas para modelos ultra-profundos exigiu inovações matemáticas severas para lidar com gargalos computacionais e com a degradação do gradiente.  Mais recentemente, a hegemonia das operações puramente convolucionais passou a ser desafiada por mecanismos de autoatenção global, redefinindo os limites do que as máquinas podem "enxergar".  Esta seção mapeia a trajetória evolutiva das arquiteturas de estado da arte em classificação de imagens, agrupando-as por seus paradigmas fundamentais de design.  A análise parte dos blocos construtivos clássicos e da consolidação da profundidade (VGG e Inception), avança pelas soluções de roteamento de gradiente (ResNet e DenseNet), explora os limites da eficiência convolucional (EfficientNet e ConvNeXt) e culmina nos modernos paradigmas baseados em atenção (Vision Transformers e Swin-T).  Para cada arquitetura, discute-se não apenas sua inovação teórica, mas principalmente sua viabilidade geométrica e computacional frente à complexidade textural do algodão. 

**2.5.1 O Legado e Consolidação (VGG e Inception)** 
A VGG estabeleceu o uso sistemático de filtros $3\times3$, demonstrando que a profundidade é o componente chave para a extração de hierarquias de características.  Embora fundamental, sua ineficiência paramétrica e o uso de camadas densas massivas servem hoje como baseline histórico.  A família Inception introduziu a ideia de processamento multiescala paralelo, capturando texturas em diferentes frequências espaciais, o que 

---

### **PÁGINA 65**

2.5. Arquiteturas de Aprendizado Profundo para Classificação de Imagens 
63 

pavimentou o caminho para redes com campos receptivos mais inteligentes. 

**2.5.1.1 A Arquitetura VGG Net** 
A VGG Net, introduzida por Simonyan e Zisserman (2014), consolidou o paradigma de que a profundidade, aliada a uma estrutura uniforme, é o fator determinante para a extração de características em larga escala.  Diferente de sua predecessora, a AlexNet, que utilizava filtros heterogêneos (como $11\times11$), a VGG propôs uma filosofia de design baseada na simplicidade: a utilização exclusiva de filtros pequenos de $3\times3$ empilhados. 

**A Geometria dos Filtros $3\times3$** 
A inovação teórica central da VGG reside na substituição de grandes núcleos convolucionais por uma pilha de núcleos menores.  Matematicamente, uma sequência de três camadas convolucionais de $3\times3$ possui o mesmo campo receptivo efetivo de uma única camada de $7\times7$. No entanto, esta abordagem oferece duas vantagens críticas para a classificação de microtexturas: 

* **Não-linearidade Adicional**: Três camadas introduzem três funções de ativação ReLU em vez de uma, permitindo que a rede aprenda funções de decisão muito mais complexas e discriminativas. 
* **Eficiência Paramétrica**: Enquanto uma camada de $7\times7$ com C canais possui $49C^{2}$ parâmetros, a pilha de três camadas de $3\times3$ utiliza apenas $27C^{2}$ parâmetros.  Esta redução de quase 45% permitiu o aumento da profundidade da rede (até 16 ou 19 camadas) sem um crescimento proibitivo do custo computacional na época. 

**Configurações e Desempenho** 
Embora existam diversas variantes (de VGG-11 a VGG-19), a VGG-16 tornou-se o padrão para tarefas de visão computacional.  Sua estrutura é composta por blocos de camadas convolucionais seguidos por max-pooling ( $(2\times2$ com stride 2), culminando em camadas totalmente conectadas (Fully Connected). 

**Limitações e Legado para a Classificação de Algodão** 

---

### **PÁGINA 66**

64 

**Tabela 3 - Comparativo Técnico das Arquiteturas VGG Net** 

| Modelo | Camadas | Parâmetros | Tamanho (MB) | Diferencial Técnico |
| :--- | :---: | :---: | :---: | :--- |
| VGG-11 | 11 | 133M | 507 | Configuração base; profundidade mínima para convergência. |
| VGG-13 | 13 | 133M | 508 | Adição de blocos convolucionais duplos no início da rede. |
| VGG-16 | 16 | 138M | 528 | Equilíbrio ideal entre profundidade e custo; arquitetura SOTA em 2014. |
| VGG-19 | 19 | 144M | 574 | Máxima capacidade de extração linear; maior custo computacional. |


Capítulo 2. Revisão da Literatura 

Apesar de sua clareza arquitetural, a VGG apresenta limitações severas em cenários modernos de produção.  A dependência de camadas totalmente conectadas massivas resulta em um alto consumo de memória (528 MB para a VGG-16).  Além disso, a ausência de conexões de salto (skip connections) torna o modelo suscetível ao desaparecimento do gradiente em profundidades maiores.  No contexto desta dissertação, a VGG-16 é analisada como um extrator de características robusto.  Sua capacidade de capturar microtexturas e sua uniformidade a tornam uma base ideal para Transfer Learning, embora sua eficiência computacional seja inferior a modelos mais recentes como a EfficientNet, um ponto crítico para a viabilidade em ambientes agrícolas de borda. 

**2.5.1.2 A Arquitetura Inception** 
Enquanto a VGG focava na profundidade linear, a arquitetura Inception (GoogLeNet), introduzida por Szegedy et al. (2015), propôs uma mudança de paradigma baseada na esparsidade e na eficiência computacional.  O objetivo central era aumentar o poder representacional da rede sem um crescimento proibitivo no número de parâmetros, utilizando para isso o "módulo Inception". 

**O Módulo Inception e o Processamento Paralelo** 
A inovação técnica fundamental da Inception v1 é o processamento multiescala simultâneo.  Em vez de selecionar um tamanho de filtro fixo, o módulo executa convoluções de 1 x 1, 3 x 3 e 5 x 5 em paralelo, concatenando seus resultados. 

---

### **PÁGINA 67**

2.5. Arquiteturas de Aprendizado Profundo para Classificação de Imagens 
65 

* **Captura Multi-resolução**: O filtro $1\times1$ foca em correlações entre canais, o $3\times3$ em características locais e o $5\times5$ em padrões globais.  Para a classificação de plumas de algodão, essa estrutura permite detectar simultaneamente fibras individuais e a morfologia geral da amostra. 
* **Redução de Dimensionalidade**: Para viabilizar o custo computacional, a Inception utiliza convoluções $1\times1$ antes dos filtros maiores para reduzir o número de canais (bottleneck), mantendo a rede leve (apenas 6,8 milhões de parâmetros na v1, contra 138 milhões da VGG-16). 

**Evolução e Fatoração (v2, v3 e Xception)** 
As versões subsequentes refinaram a arquitetura através de princípios de design heurístico: 

* **Fatoração de Convoluções $(v2/v3)$**: Seguindo o princípio da eficiência, kernels grandes $(5\times5)$ foram substituídos por dois de $3\times3$. Mais além, convoluções $n\times n$ foram fatoradas em sequências assimétricas de $1\times n~e~n\times1$ (ex: $1\times7~e~7\times1)$ reduzindo drasticamente o custo computacional sem perda de campo receptivo. 
* **Xception (Extreme Inception)**: Proposta por Chollet (2017), esta arquitetura leva o desacoplamento ao limite, utilizando Convoluções Separáveis em Profundidade (Depthwise Separable Convolutions).  Ela separa totalmente a busca por correlações espaciais das correlações entre canais, sendo hoje um dos backbones mais eficazes para tarefas que exigem alta precisão em texturas finas. 

**2.5.2 Robustez e Fluxo de Gradiente (ResNet e DenseNet)** 
A segunda geração resolveu o problema do desaparecimento do gradiente em redes profundas. 

* **Aprendizado Residual (ResNet)**: Ao introduzir o mapeamento residual, a ResNet He et al. (2016) permitiu o treinamento de centenas de camadas. Para o algodão, isso garante que características fundamentais da fibra não sejam degradadas ao longo da propagação. 

---

### **PÁGINA 68**

66 
Capítulo 2. Revisão da Literatura 

**Tabela 4 - Evolução Cronológica e Inovações da Família Inception** 

| Versão | Inovação Principal | Impacto Acadêmico / Técnico |
| :--- | :--- | :--- |
| Inception v1 | Módulo Inception | Introdução do processamento multiescala paralelo e redução de parâmetros. |
| Inception v2 | Batch Normalization | Estabilização do treinamento e aceleração da convergência. |
| Inception v3 | Fatoração Assimétrica | Substituição de filtros $n\times n$ por $1\times n~e~n\times1$; redução de gargalos. |
| Inception v4 | Inception-ResNet | Integração de conexões residuais para treinamento de redes ultra-profundas. |
| Xception | Depthwise Separable | Desacoplamento total de correlações espaciais e de canal; máxima eficiência. |



* **Conectividade Densa (DenseNet)**: Huang et al. (2017) maximizaram a reutilização de características (feature reuse).  Como a pluma de algodão possui padrões repetitivos e sutis, a capacidade da DenseNet de manter os mapas de características de baixo nível acessíveis a todas as camadas subsequentes é uma vantagem teórica na preservação de microtexturas. 

**2.5.2.1 A Arquitetura ResNet** 
A introdução das Redes Residuais (ResNet) por He et al. (2016) marcou um ponto de inflexão na visão computacional ao solucionar o problema da degradação em redes ultra-profundas.  Antes da ResNet, adicionar camadas a uma CNN resultava paradoxalmente em maiores erros de treinamento, devido à dificuldade dos otimizadores em convergir para mapeamentos de identidade em arquiteturas puramente sequenciais. 

**O Mecanismo de Aprendizagem Residual** 
A premissa da ResNet é que é intrinsecamente mais fácil otimizar um resíduo do que um mapeamento direto.  Em vez de forçar as camadas a aprenderem uma função desejada $H(x)$, a arquitetura as projeta para aprender a função residual $F(x)=H(x)-x$. O resultado final é obtido pela soma: 
$H(x)=F(x)+x$ 
Essa operação é viabilizada pelas conexões de atalho (skip connections), que permitem 

---

### **PÁGINA 69**

2.5. Arquiteturas de Aprendizado Profundo para Classificação de Imagens 
67 

que o sinal do gradiente flua diretamente pelas camadas sem atenuação.  No reconhecimento de plumas, onde sutilezas de cor e textura podem ser "lavadas" por muitas operações convolucionais, o caminho de identidade garante que a informação visual bruta persista até as camadas de decisão final. 

**Evolução dos Blocos: Basic vs. Bottleneck** 
A arquitetura evolui em complexidade de acordo com a profundidade desejada, utilizando dois tipos de blocos fundamentais: 

* **BasicBlock (ResNet-18/34)**: Composto por duas convoluções $3\times3$. É robusto e eficaz para extração de texturas em datasets de médio porte. 
* **Bottleneck (ResNet-50/101/152)**: Utiliza uma estratégia de compressão e expansão com convoluções $l\times1$ envolvendo uma convolução $3\times3$. Este design reduz drasticamente o número de parâmetros e GFLOPs, permitindo redes muito mais profundas (como a ResNet-101) com um custo computacional menor que a VGG-16. 

**Tabela 5 - Especificações Técnicas da Família ResNet** 

| Modelo | Bloco | Parâmetros | GFLOPs | Cenário de Uso |
| :--- | :---: | :---: | :---: | :--- |
| ResNet-18 | Basic | 11.7M | 1.8 | Dispositivos móveis e edge. |
| ResNet-34 | Basic | 21.8M | 3.6 | Baselines de classificação de textura. |
| ResNet-50 | Bottleneck | 25.6M | 3.8 | Padrão industrial para transfer learning. |
| ResNet-101 | Bottleneck | 44.5M | 7.6 | Alta precisão em imagens complexas. |



**2.5.2.2 A Arquitetura DenseNet** 
A arquitetura DenseNet (Densely Connected Convolutional Network), proposta por Huang et al. (2017), representa uma evolução radical na topologia das CNNs.  Enquanto a ResNet utiliza conexões de salto via soma, a DenseNet estabelece que cada camada deve ser conectada a todas as outras camadas subsequentes dentro de um bloco denso através de operações de concatenação. 

---

### **PÁGINA 70**

68 
**O Paradigma da Conectividade Densa** 
Capítulo 2. Revisão da Literatura 

Matematicamente, em uma rede tradicional, a camada I recebe a saída da camada anterior $(l-1)$.  Na DenseNet, a camada l recebe como entrada todos os mapas de características das camadas precedentes: 
$x_{l}=H_{l}([x_{0},x_{1},...,x_{l-1}])$ 
Onde $[x_{0},x_{1},...,x_{l-1}]$ representa a concatenação de todas as representações aprendidas anteriormente.  Esta abordagem oferece três vantagens teóricas para a classificação de fibras e texturas: 

* **Reuso de Características**: Filtros fundamentais (como detectores de bordas e fibras individuais) aprendidos no início do bloco são passados adiante de forma intacta.  Isso evita que a rede precise "reaprender" padrões básicos em camadas profundas. 
* **Eficiência de Parâmetros**: Como o conhecimento é compartilhado, as camadas individuais da DenseNet podem ser muito "estreitas" (produzindo apenas 12 ou 32 novos filtros por camada), resultando em modelos significativamente mais leves que as ResNets equivalentes. 
* **Supervisão Profunda Implícita**: Cada camada recebe um sinal de gradiente direto da função de perda através das múltiplas conexões, o que estabiliza o treinamento e reduz o risco de desaparecimento do gradiente. 

**Taxa de Crescimento e Camadas de Transição** 
Dois componentes definem a dinâmica da DenseNet: 

* **Taxa de Crescimento (k)**: É o hiperparâmetro que regula quantos novos mapas de características cada camada adiciona ao "estado global" da rede.  Mesmo com um k pequeno (ex: $k=32$), a densidade de conexões garante um poder representacional elevado. 

---

### **PÁGINA 71**

2.5. Arquiteturas de Aprendizado Profundo para Classificação de Imagens 
69 

* **Camadas de Transição**: Posicionadas entre blocos densos, estas camadas realizam a redução espacial (via pooling) e a compressão de canais (via convolução $1\times1$), controlando a explosão dimensional decorrente das concatenações. 

Para o problema de 10 classes de algodão, a DenseNet destaca-se pela sua robustez em datasets limitados.  A literatura indica que a conectividade densa atua como um regularizador natural, reduzindo o overfitting em tarefas onde a variância inter-classe é sutil, tornando-a uma das arquiteturas mais eficazes para o reconhecimento de texturas fibrosas e orgânicas. 

**Tabela 6 - Variações da Arquitetura DenseNet-BC e Cenários de Aplicação** 

| Modelo | Camadas | Parâmetros | GFLOPs | Cenário de Uso Sugerido |
| :--- | :---: | :---: | :---: | :--- |
| DenseNet-121 | 121 | 8.0M | 2.8 | Baseline para classificação de texturas; ideal para prototipagem rápida e transfer learning. |
| DenseNet-169 | 169 | 14.1M | 3.4 | Cenários de média complexidade; equilíbrio entre custo computacional e acurácia. |
| DenseNet-201 | 201 | 20.0M | 4.3 | Padrão para imagens médicas e biológicas; alta capacidade de extração de microtexturas. |
| DenseNet-264 | 264 | 33.3M | 5.8 | Pesquisas de alta precisão; cenários onde o volume de dados permite redes ultra-profundas. |



**2.5.3 Modernização e Eficiência Convolucional (Efficient Net e ConvNeXt)** 
Enquanto a VGG focava em profundidade e a Inception em largura, a EfficientNet introduz a harmonia entre as dimensões.  No caso específico das plumas de algodão, essa arquitetura é vital porque permite processar imagens de alta resolução (essencial para ver fibras individuais) sem explodir o custo computacional, graças ao escalonamento composto. 

---

### **PÁGINA 72**

70 
Capítulo 2. Revisão da Literatura 

**2.5.3.1 A Arquitetura Efficient Net** 
A família EfficientNet, introduzida por Tan e Le (2019), redefiniu o design de CNNs ao propor que o aumento da capacidade de um modelo não deve ser arbitrário.  A inovação central reside no Escalonamento Composto (Compound Scaling), um método que equilibra simultaneamente a profundidade, a largura e a resolução da rede. 

**O Princípio do Escalonamento Composto** 
Diferente de arquiteturas anteriores que escalonavam apenas uma dimensão (ex: a profundidade na ResNet), a EfficientNet utiliza um coeficiente composto coordenar o crescimento estruturado da rede.  Matematicamente, as dimensões são definidas como: 

* **Profundidade (d)**: $d=\alpha^{\phi}$ 
* **Largura (w)**: $w=\beta^{\phi}$ 
* **Resolução (r)**: $r=\gamma^{\phi}$ 

Sob a restrição de que $\alpha\cdot\beta^{2}\cdot\gamma^{2}\approx2$, o aumento total de FLOPs é de aproximadamente $2^{\phi}$.  Para a classificação de algodão, isso garante que se aumentarmos a resolução para detectar micro-impurezas, a rede automaticamente aumentará sua profundidade (para expandir o campo receptivo) e sua largura (para processar a densidade maior de dados), mantendo a eficiência operacional. 

**Arquitetura Base: MBConv e Squeeze-and-Excitation** 
A EfficientNet-BO (o modelo base) foi desenvolvida através de Busca de Arquitetura Neural (NAS).  Seu bloco fundamental é o MBConv (Mobile Inverted Bottleneck), que utiliza: 

* **Convoluções Separáveis em Profundidade**: Reduzem drasticamente o número de operações em comparação com as convoluções densas. 
* **Mecanismo Squeeze-and-Excitation (SE)**: Uma forma de atenção que recalibra a importância de cada canal, permitindo que o modelo foque em características discriminativas da fibra. 

---

### **PÁGINA 73**

2.5. Arquiteturas de Aprendizado Profundo para Classificação de Imagens 
71 

* **Ativação Swish**: Uma função suave e não-monotônica $(f(x)=x\cdot\sigma(x))$ que auxilia na propagação de gradientes em redes profundas. 

**Efficient NetV2: Fused-MBConv e Velocidade** 
A evolução para a Tan e Le (2021) resolveu gargalos de hardware da primeira versão.  Nas camadas iniciais, onde a convolução depthwise é menos eficiente em aceleradores modernos (GPUs), a V2 introduz o Fused-MBConv, que substitui a expansão $1\times1$ e a convolução $3\times3$ por uma única convolução padrão $3\times3$. Isso resulta em um treinamento até 11 vezes mais rápido e modelos ainda menores. 

**Tabela 7 - Comparativo de Eficiência: EfficientNet vs. Família ResNet** 

| Modelo | Precisão Top-1 | Parâmetros | FLOPs | Vantagem na Produção |
| :--- | :---: | :---: | :---: | :--- |
| ResNet-50 | 76,3% | 26,0M | 4,1B | Padrão de mercado, porém pesado. |
| EfficientNet-BO | 77,1% | 5,3M | 0,39B | 5x menor e superior em acurácia. |
| EfficientNet-B4 | 82,9% | 19,0M | 4,2B | Similar à ResNet-50 em custo, mas +6% de acurácia. |
| EfficientNetV2-S | 83,9% | 22,0M | 8,8B | Otimizada para velocidade de treino e inferência. |



**2.5.3.2 A Arquitetura ConvNeXt** 
A arquitetura ConvNeXt, proposta por Liu et al. (2022), surgiu como uma resposta direta à ascensão dos Vision Transformers (ViTs).  O objetivo do projeto foi investigar se a superioridade dos Transformers residia no mecanismo de atenção ou em avanços de design macroarquitetural.  Através de um processo de "modernização" de uma ResNet padrão, os autores provaram que as redes puramente convolucionais ainda podem superar os Transformers em precisão e eficiência. 

**O Processo de Modernização e Design** 
A ConvNeXt integra lições de design dos Transformers em uma estrutura convolucional, adotando mudanças fundamentais: 

---

### **PÁGINA 74**

72 
Capítulo 2. Revisão da Literatura 

* **Macro Design**: Substituição do stem inicial por uma camada de "patchify" (convolução $4\times4$ com stride 4) e ajuste na distribuição de blocos por estágio (razão 1:1:3:1), emulando a hierarquia do Swin Transformer. 
* **Gargalo Invertido (Inverted Bottleneck)**: Adoção de uma estrutura onde a dimensão oculta é expandida, permitindo que as convoluções espaciais operem em uma dimensionalidade maior antes da projeção de canais. 
* **Aumento do Campo Receptivo**: Substituição de filtros $3\times3$ por kernels de $7\times7$. Para o algodão, esse campo receptivo ampliado é crucial para capturar a continuidade de fibras longas e a morfologia de impurezas maiores dispersas na pluma. 

**Micro Design: Simplicidade e Estabilidade** 
A rede simplifica os componentes internos para melhorar o fluxo de gradiente e a estabilidade: 

* **Ativação e Normalização**: Substituição de ReLU por GELU e de BatchNorm por LayerNorm, aplicadas de forma mais esparsa (apenas uma vez por bloco), reduzindo a redundância e o custo computacional. 
* **Global Response Normalization (GRN)**: Na versão V2, foi introduzida a camada GRN para evitar o "colapso de atributos" em treinamentos auto-supervisionados, aumentando a competição entre canais e garantindo que cada filtro aprenda características únicas da textura da pluma. 

**Tabela 8 - Comparativo de Performance e Eficiência: ConvNeXt vs. Swin Transformer** 

| Arquitetura | Modelo | Parâmetros | FLOPs | Acurácia (IN-1K) |
| :--- | :--- | :---: | :---: | :---: |
| Transformer | Swin-T | 28M | 4,5G | 81,3% |
| Convolucional | ConvNeXt-T | 28M | 4,5G | 82,1% |
| Transformer | Swin-B | 88M | 15,4G | 83,5% |
| Convolucional | ConvNeXt-B | 89M | 15,4G | 83,8% |



---

### **PÁGINA 75**

2.5. Arquiteturas de Aprendizado Profundo para Classificação de Imagens 
73 

**2.5.4 Paradigmas Baseados em Atenção (ViT e Swin Transformer)** 

**2.5.4.1 A Arquitetura ViT** 
A introdução do Vision Transformer (ViT) por Dosovitskiy et al. (2021) desafiou a soberania das convoluções ao transpor a arquitetura Transformer, originalmente projetada para Processamento de Linguagem Natural (PLN), diretamente para o domínio da visão computacional.  A premissa fundamental é tratar uma imagem não como uma grade de pixels, mas como uma sequência de "palavras" visuais (tokens). 

**Tokenização e Mecanismo de Autoatenção** 
O processo operacional do ViT diverge drasticamente das CNNs: 

* **Patchification**: A imagem é dividida em retalhos (patches) fixos (ex: $16\times16$ pixels).  Cada patch é achatado e projetado linearmente para um espaço de embedding. 
* **Embeddings Posicionais**: Como o Transformer processa os dados em paralelo e carece de viés indutivo de localidade, são adicionados vetores posicionais para que a rede "aprenda" a geografia da imagem. 
* **Self-Attention Global**: Diferente das convoluções, que possuem campo receptivo limitado ao tamanho do kernel, a autoatenção permite que cada patch interaja com todos os outros desde a primeira camada.  Isso permite ao modelo correlacionar fibras em extremidades opostas da imagem, capturando uma semântica global que define a qualidade da pluma. 

**Vieses Indutivos e a Necessidade de Dados** 
Diferente das CNNs, que possuem "vieses indutivos" (localidade e invariância à translação) embutidos em sua matemática, o ViT é uma arquitetura de "propósito geral".  Em datasets pequenos, o ViT tende a apresentar desempenho inferior às CNNs, pois carece da noção inata de pixels vizinhos.  Entretanto, ao ser pré-treinado em volumes massivos de dados (como o ImageNet-21k), sua capacidade de generalização supera as redes convolucionais.  Para a aplicação no escopo desta dissertação, onde o dataset de 

---

### **PÁGINA 76**

74 
Capítulo 2. Revisão da Literatura 

plumas de algodão é naturalmente limitado, a utilização do ViT torna-se estritamente dependente de técnicas de Transfer Learning para herdar a compreensão espacial dos pré-treinos. 

**Tabela 9 - Comparativo de Paradigmas: CNN vs. Vision Transformer (ViT)** 

| Característica | Convolucionais (CNN) | Vision Transformer (ViT) |
| :--- | :--- | :--- |
| Unidade de Processamento | Janelas locais (Pixels) | Patches de imagem (Tokens) |
| Campo Receptivo | Cresce com a profundidade | Global desde a primeira camada |
| Viés Indutivo | Localidade e Translação (Fortes) | Fraco (Aprendido via dados) |
| Complexidade Espacial | Linear $O(N)$ | Quadrática $O(N^{2})$ |
| Escalabilidade | Satura em grandes datasets | Melhora com o aumento de dados |



**2.5.4.2 A Arquitetura Swin-T** 
O Swin Transformer (Shifted Window Transformer), proposto por Liu et al. (2021), resolveu as duas principais limitações do ViT original: a complexidade computacional quadrática e a ausência de uma estrutura multiescala.  Ele reintroduziu a pirâmide de características (típica das CNNs) no paradigma da autoatenção. 

**Hierarquia e Complexidade Linear** 
Diferente do ViT, que mantém uma resolução única (isotrópica) em toda a rede, o Swin utiliza camadas de Patch Merging para reduzir a resolução espacial enquanto aumenta a profundidade dos canais.  Para reduzir o custo computacional, o Swin divide a imagem em janelas locais $(ex:7\times7$ patches) e calcula a atenção apenas dentro dessas janelas.  Isso reduz a complexidade de $O(N^{2})$ para $O(N)$, permitindo o processamento de imagens em alta definição. 

**O Mecanismo de Janelas Deslocadas (SW-MSA)** 
Para garantir que a rede não fique limitada a visões isoladas das janelas, o Swin alterna entre o particionamento regular e o particionamento deslocado (Shifted Windows).  O deslocamento da grade de janelas entre camadas consecutivas cria conexões entre patches que pertenciam a janelas diferentes na camada anterior.  Esse mecanismo permite que o modelo correlacione fibras ou impurezas que cruzam as fronteiras das janelas, garantindo uma compreensão estrutural contínua da 

---

### **PÁGINA 77**

2.6. Síntese da Literatura e Lacunas da Pesquisa 
75 

pluma de algodão, combinando o melhor das hierarquias convolucionais com a robustez da atenção global. 

**Tabela 10 - Especificações Técnicas e Aplicações das Variantes Swin Transformer** 

| Modelo | Parâmetros | GFLOPS | Top-1 Acc* | Cenário de Uso na Classificação de Algodão |
| :--- | :---: | :---: | :---: | :--- |
| Swin-T (Tiny) | 28M | 4,5 | 81,3% | Classificação em dispositivos de borda ou triagem preliminar de amostras. |
| Swin-S (Small) | 50M | 8,7 | 83,0% | Equilíbrio entre custo e precisão para monitoramento de linha de produção. |
| Swin-B (Base) | 88M | 15,4 | 83,5% | Padrão para pesquisa acadêmica; alta capacidade de distinção entre classes similares. |
| Swin-L (Large) | 197M | 34,5 | 87,3%** | Máximo desempenho para análise laboratorial de granulação fina (fine-grained). |
| SwinV2-G (Giant) | 3B | 650+ | 90,1% | Modelos de fundação; voltado para benchmarking e pré-treino massivo. |


*Acurácia medida no ImageNet-1K. **Treinado com ImageNet-22K. 

**2.6 Síntese da Literatura e Lacunas da Pesquisa** 
A revisão bibliográfica apresentada neste capítulo evidencia que a classificação da qualidade da pluma de algodão representa um gargalo crítico na cadeia produtiva.  Historicamente dependente da avaliação humana subjetiva ou de instrumentos laboratoriais de alto custo, como o HVI, o setor encontra na Visão Computacional e no Aprendizado Profundo as soluções tecnológicas mais viáveis para a automação objetiva e em larga escala. 

O estado da arte revela uma transição clara nas abordagens de extração de características visuais.  A trajetória das Redes Neurais Convolucionais (CNNs), culminando em arquiteturas modernas como EfficientNet e ConvNeXt, estabeleceu um patamar elevado de eficiência e robustez espacial.  Essas arquiteturas provaram ser particularmente eficazes na extração de microtexturas e na detecção de impurezas (trash), características 

---

### **PÁGINA 78**

76 
Capítulo 2. Revisão da Literatura 

fundamentais para a análise de fibras orgânicas. Por outro lado, a ascensão dos modelos baseados em atenção, representados pelo ViT e pelo Swin Transformer, introduziu a capacidade de modelar dependências globais e relações de longa distância na imagem, permitindo uma análise estrutural que antes era limitada pelo campo receptivo fixo das convoluções. 

A literatura converge para a premissa de que o desempenho superior em tarefas de classificação industrial não advém apenas da topologia da rede, mas da simbiose entre arquitetura, técnicas de Transfer Learning e receitas modernas de treinamento incluindo otimizadores desacoplados (como AdamW) e técnicas de regularização. 

Apesar da consolidação dessas tecnologias em domínios genéricos de visão computacional, a literatura focada especificamente na classificação automatizada da pluma de algodão ainda é fragmentada.  A ausência de uma comparação sistemática e direta entre as arquiteturas convolucionais modernas e os Transformers hierárquicos neste contexto agroindustrial revela as seguintes lacunas de pesquisa, as quais justificam o desenvolvimento desta dissertação: 

* **Escassez de Dados em Domínios Específicos e Dependência de Pré-treino**: A maioria dos estudos que validam a superioridade dos Vision Transformers assume a disponibilidade de datasets massivos (como ImageNet-21k ou JFT-300M).  A eficácia e a capacidade de convergência dessas arquiteturas especialmente o Swin Transformer em datasets restritos de pluma de algodão, onde a anotação de especialistas é custosa e escassa, ainda carecem de validação empírica profunda. 
* **Similaridade Inter-classes Extrema (Fine-Grained Classification)**: A distinção entre as classes comerciais de pluma envolve variações cromáticas sutis (reflectância e amarelamento) e a identificação de defeitos milimétricos (como neps).  A literatura atual não explorou exaustivamente como os diferentes mecanismos de extração (convolução local vs. atenção global) lidam com a confusão entre classes vizinhas no espectro de qualidade, caracterizando este problema como um autêntico desafio de classificação de granulação fina (fine-grained). 
* **Robustez sob Condições Operacionais Reais**: A literatura frequentemente avalia modelos em condições laboratoriais altamente controladas. Há uma lacuna sig- 

---

### **PÁGINA 79**

2.6. Síntese da Literatura e Lacunas da Pesquisa 
77 

nificativa quanto à resiliência de arquiteturas estado da arte frente às variações de iluminação, presença de sombras e heterogeneidade na preparação da amostra, fatores típicos da esteira de processamento de algodão. 
* **Trade-off entre Acurácia e Latência para Edge Computing**: Embora modelos como a EfficientNet ofereçam eficiência teórica, o impacto real do custo computacional de mecanismos de autoatenção (Swin) na latência de inferência é pouco documentado na área agrícola.  Para que sistemas de classificação operem em tempo real no setor produtivo (Edge Computing), é imperativo estabelecer benchmarks que comparem não apenas o F1-Score, mas o custo paramétrico e a velocidade de inferência (FLOPs) em hardwares limitados. 

Em suma, esta pesquisa propõe-se a preencher essas lacunas ao avaliar, de forma inédita e comparativa, o desempenho das principais arquiteturas de Deep Learning aplicadas à classificação do algodão.  O objetivo final é fornecer uma análise empírica robusta que subsidie decisões técnicas no setor produtivo, buscando o equilíbrio ótimo entre o rigor acadêmico da classificação de alta complexidade e a viabilidade técnica de implementação industrial. 

---

### **PÁGINA 81**

79 
CAPÍTULO 3 
METODOLOGIA 

Nesta seção, são detalhados os procedimentos e as ferramentas desenvolvidas  para a aquisição de dados, que constituem a base para o treinamento e validação do sistema de visão computacional proposto.  A metodologia foi dividida em duas etapas principais: (1) o desenvolvimento de um sistema de hardware para aquisição padronizada de imagens, doravante denominado "Caixa de Captura";  (2) o estabelecimento de um protocolo rigoroso para a coleta das amostras de algodão em ambiente industrial;  o (3) pré-processamento e preparação dos dados; e (4) o ambiente de treinamento e avaliação dos modelos.  Cada etapa foi projetada com foco na robustez, padronização e reprodutibilidade dos resultados, pilares fundamentais para a validação científica do estudo. 

**3.1 Desenvolvimento do Sistema de Aquisição de Dados (Caixa de Captura)** 
A estrutura física consiste em uma cabine retangular fechada, construída em MDF, com dimensões de $40\times40\times60$ cm (largura, profundidade e altura).  O interior foi revestido com material branco e opaco, com o duplo objetivo de bloquear a luz externa e evitar a contaminação colorimétrica por reflexos das paredes.  O equipamento possui um compartimento superior isolado para acomodar as fontes de alimentação e a eletrônica, garantindo que a área inferior de captura fique livre de cabos e sombras indesejadas. 

---

### **PÁGINA 82**

80 
Capítulo 3. Metodologia 

dispositivo de captura foi montado centralizado no teto da estrutura, a uma distância focal aproximada de 40 cm da base onde as amostras são introduzidas (através de uma abertura frontal), garantindo um enquadramento constante.  Um esquema técnico detalhado do equipamento é apresentado na Figura 4. 

O sistema de iluminação foi projetado estrategicamente para lidar com os desafios ópticos da pluma de algodão.  Utilizaram-se duas matrizes de fitas de LED de alta intensidade $(2000~lm/m)$ posicionadas na parte superior, fornecendo luz difusa para minimizar a geração de sombras profundas e brilhos especulares na fibra.  O diferencial do sistema reside na alternância de temperaturas de cor: LEDs de 2800K (luz "quente") foram empregados para realçar a reflectância e o amarelamento natural da fibra (+b), enquanto LEDs de 6000K (luz "fria", próxima à luz do dia) maximizaram o contraste morfólogico das impurezas superficiais.  O protocolo de captura gerou três imagens por amostra: apenas 2800K, apenas 6000K, e ambas simultaneamente, compondo um dataset multiespectral robusto para o treinamento das redes. 

O módulo de aquisição baseou-se em um sensor CMOS operando a uma resolução de $1280\times720$ pixels.  Optou-se por esta tecnologia devido à sua alta viabilidade de implantação em esteiras industriais, unindo baixo custo e baixo consumo energético, conforme fundamentado na revisão de literatura.  O sensor foi acoplado a uma lente ultra-angular com campo de visão (Field of View - FOV) de $140^{\circ}$, o que permitiu maximizar a área de amostragem do algodão capturada na base do equipamento sem a necessidade de elevar a altura da caixa.  As distorções geométricas inerentes a lentes de grande angulação foram amenizadas pelo posicionamento centralizado da amostra. 

**3.2 Protocolo de Coleta e Amostragem** 
O processo de aquisição de dados foi conduzido em ambiente industrial, diretamente nas instalações de uma unidade de beneficiamento de algodão (algodoeira) localizada no município de Sapezal, Mato Grosso, um dos principais polos produtivos do cerrado brasileiro.  A coleta ocorreu durante a safra de 2023/2024, totalizando 3.852 imagens brutas, sendo 1.284 para cada uma das três configurações de iluminação (2800K, 6000K e ambas). 

---

### **PÁGINA 83**

3.2. Protocolo de Coleta e Amostragem 
40cm 
81 

**Figura 4 - Equipamento de coleta de amostras** 
(Legendas da figura: Tampo superior; Espaço dedicado apra acomodar as fontes de alimentação das fitas LED e Câmera; Fitas LED; Quatro fitas de LED de 50cm (Duas de 2800K e Duas de 6000K) coladas; Abertura central para a câmera; Abertura de cerca de 1cm de diâmetro para acomodar a câmera; Abertura frontal para manipular a amostra; Onde a amostra é introduzida e manipulada para a captura das imagens.) 

se um protocolo padronizado de captura, desenhado não apenas para gerar as imagens, mas para assegurar o vínculo exato com os laudos de qualidade oficiais (o ground truth dos modelos).  O fluxo de trabalho operacionalizou-se nas seguintes etapas, ilustradas no fluxograma da Figura 5: 

1.  **Identificação e Rastreabilidade**: O operador inicia o ciclo utilizando um leitor de código de barras para escanear a etiqueta universal da amostra.  Este procedimento vincula automaticamente o ID único do fardo, o lote de processamento e a origem da pluma aos metadados dos arquivos de imagem que serão gerados, eliminando o risco de erros de digitação e garantindo o cruzamento futuro com os dados de classificação oficial (HVI/Visual). 
2.  **Posicionamento Padronizado**: A porção de algodão é extraída de sua embalagem  e alocada no centro da base do equipamento de captura.  O operador realiza a padronização manual do formato da amostra, acomodando a superfície da pluma para garantir a máxima planicidade possível.  Este rigor na modelagem física da amostra é crucial para evitar oclusões severas, minimizar a formação de sombras irregulares e assegurar que a área exposta à lente seja representativa da densidade real do fardo. 
3.  **Captura Multiespectral Automatizada**: Através da interface do software de controle, a sequência de aquisição é disparada.  O operador orquestra manualmente o acionamento dos LEDs e a captura das imagens em três condições sequenciais: iluminação exclusiva de 2800K (AM), iluminação exclusiva de 6000K (BR) e iluminação combinada (AB).  As três matrizes de pixels resultantes são salvas em formato sem perdas e atreladas ao ID escaneado no Passo 1. 
4.  **Encaminhamento para Classificação Oficial**: Concluída a aquisição visual, a amostra física é imediatamente reconduzida ao fluxo normal da algodoeira, sendo destinada ao laboratório para a classificação comercial humana.  Esse passo é o que permite a rotulagem supervisionada do dataset nas etapas posteriores da pesquisa. 

---

### **PÁGINA 84**

82 
Capítulo 3. Metodologia 

**Figura 5 - Fluxograma do processo de coleta de amostras e aquisição de imagens.** 
(Etapas do fluxograma: Início; 1. Ler código de barras da amostra; 2. Posicionar amostra na Caixa de Captura; 3. Iniciar processo de captura no software; 4. Capturar imagens: Iluminação AM (2800K), Iluminação BR (6000K), Iluminação AB (Ambas); 5. Salvar as três imagens associadas ao ID; 6. Remover amostra da caixa; Próxima Amostra.) 

A execução rigorosa deste protocolo resultou em um conjunto de dados primário robusto, onde cada amostra física originou múltiplas imagens sob diferentes condições espectrais, todas rigorosamente rastreáveis até o seu laudo de qualidade comercial. 

---

### **PÁGINA 85**

3.3. Pré-processamento e preparação dos dados 
83 

**3.3 Pré-processamento e preparação dos dados** 
A etapa de preparação de dados é um pilar fundamental em projetos de visão computacional, sendo decisiva para a capacidade de generalização das redes neurais.  O objetivo deste processo foi transformar as 3.852 imagens brutas, capturadas no ambiente industrial, em tensores padronizados, isolando a pluma de algodão de qualquer ruído de fundo.  Para garantir a rastreabilidade, reprodutibilidade e governança do experimento, a arquitetura do pipeline de dados foi inspirada no paradigma de medalhas (Medallion Architecture), popularizado pela plataforma Databricks.  Dessa forma, o fluxo foi estruturado em três estágios lógicos de refinamento: dados brutos e imutáveis (Bronze), dados segmentados e padronizados (Silver), e tensores finalizados para treinamento (Gold). 

**3.3.1 Segmentação Semântica e Remoção de Fundo** 
A primeira transformação aplicada visou isolar estritamente a Região de Interesse (Area of Interest - AOI), removendo as paredes da caixa de captura e a etiqueta amarela de identificação.  Para garantir o alinhamento perfeito, a imagem capturada sob iluminação combinada (AB - 2800K e 6000K) foi eleita como referência geométrica (master image) para cada amostra, por oferecer o maior contraste global.  Em vez de técnicas clássicas de limiarização, que sofrem com as variações de iluminação e sombras inerentes à textura do algodão, optou-se por uma abordagem baseada em aprendizado profundo para a remoção do fundo.  Utilizou-se o modelo BiRefNet (Bilateral Reference Network), uma arquitetura de estado da arte para Segmentação Dicotômica de Imagens (DIS).  O BiRefNet foi escolhido por sua excepcional capacidade de preservar contornos complexos e felpudos, gerando uma máscara binária de alta precisão para a massa de algodão.  Simultaneamente, para eliminar a etiqueta de rastreabilidade do campo de visão, realizou-se uma segmentação por cor.  A imagem foi convertida para o espaço de cores HSV (Hue, Saturation, Value), permitindo a criação de uma máscara específica para os tons de amarelo da etiqueta, independentemente da intensidade luminosa.  Operações morfológicas de fechamento (closing) e dilatação foram aplicadas para garantir a cobertura 

---

### **PÁGINA 86**

84 
Capítulo 3. Metodologia 

total da área impressa. Por fim, a máscara da etiqueta foi subtraída da máscara principal do BiRefNet, resultando no isolamento puro da pluma de algodão.  A evolução visual desse processo de filtragem morfológica - partindo da imagem bruta até a consolidação da máscara final é ilustrada sequencialmente nas Figuras 6a até 6d. 

**3.3.2 Cálculo do Maior Retângulo Interno (Largest Interior Rectangle)** 
Com a pluma isolada, o passo seguinte foi realizar o corte (crop).  O uso de uma caixa delimitadora tradicional (Bounding Box) foi descartado, pois as bordas orgânicas da amostra fariam com que pixels pretos (do fundo já removido) fossem incluídos no corte, introduzindo ruído no treinamento.  Para solucionar este problema, empregou-se o algoritmo do Maior Retângulo Interno (Largest Interior Rectangle - LIR).  Dado um polígono irregular (o contorno da máscara do algodão), o LIR busca encontrar o retângulo A de área máxima que esteja estritamente contido dentro de $\phi$, tal que $\mathbb{R}\subseteq\mathbb{P}$. Este é um problema clássico de geometria computacional, resolvido através da busca do maior sub-histograma em matrizes binárias.  As coordenadas retangulares $(x_{min},y_{min},x_{max},y_{max})$ obtidas pelo LIR na imagem mestre (AB) definiram a AOI definitiva, conforme demonstrado nas Figuras 6e e 6f.  Esta mesma janela de corte foi então aplicada de forma determinística às imagens correspondentes sob luz quente (AM) e luz fria (BR).  Isso garantiu que, para uma mesma amostra física, as três variações de iluminação possuíssem uma amostragem de pixels espacialmente idêntica. 

**3.3.3 Ladrilhamento (Patching) e Estruturação do Dataset Final** 
Arquiteturas modernas de Redes Neurais Convolucionais e Transformers tipicamente exigem tensores de entrada quadrados e de dimensões fixas.  Como as AOIs geradas pelo LIR possuíam dimensões variadas e resoluções superiores às suportadas pelas redes, adotou-se a estratégia de ladrilhamento (tiling ou patching).  Cada recorte retangular foi subdividido em múltiplos patches quadrados de 

---

### **PÁGINA 87**

3.3. Pré-processamento e preparação dos dados 
85 

$256\times256$ pixels. Para evitar a perda de informações nas bordas e nas transições de textura, implementou-se uma taxa de sobreposição (overlap) entre os patches adjacentes.  Esta técnica, além de garantir a cobertura integral da AOI, atua como uma forma robusta de aumento de dados (Data Augmentation), multiplicando significativamente o volume do conjunto de treinamento e expondo o modelo a uma maior variabilidade de microtexturas locais da fibra.  Ao final deste processo (Camada Gold), os patches foram convertidos para matrizes numéricas e vinculados de volta ao seu ground truth (a classe de qualidade oficial obtida no laboratório).  Os dados foram então particionados de forma estratificada nos subconjuntos de Treinamento, Validação e Teste, formatados de acordo com os padrões dos DataLoaders do framework PyTorch, consolidando um dataset otimizado e pronto para a experimentação arquitetural. 

**Figura 6 - Etapas sequenciais do pipeline de pré-processamento visual: segmentação semântica, operações morfológicas para exclusão de ruídos e definição geométrica da Região de Interesse (AOI).** 
(Sublegendas: (a) Imagem Bruta (Original); (b) Máscara BiRefNet; (c) Máscara da Etiqueta (HSV); (d) Máscara Final (Subtração); (e) Maior Retângulo Interno (LIR); (f) Recorte Final (AOI).) 

---

### **PÁGINA 88**

84 
Capítulo 3. Metodologia 

**3.4 Construção do Conjunto de Dados, Rotulagem e Balanceamento de Classes** 
O escopo desta pesquisa foi direcionado para as 10 classes comerciais de algodão em pluma mais prevalentes na produção local, conforme os padrões oficiais de classificação estabelecidos pela Instrução Normativa nº 24/2016 do Ministério da Agricultura, Pecuária e Abastecimento (MAPA).  A rotulagem de cada amostra (ground truth) foi estabelecida com base no laudo de classificação visual humano adotado pela própria algodoeira parceira.  Esse procedimento assegurou que os rótulos atribuídos aos tensores de imagem correspondessem estritamente aos parâmetros utilizados nas operações reais de comercialização, conferindo validade industrial ao problema de classificação. 

Após a etapa de pré-processamento (Camada Gold), o dataset final consolidou-se em 7.908 recortes (patches) de imagem.  Para a condução rigorosa dos experimentos, realizou-se a divisão do conjunto de dados utilizando uma proporção de 80% para Treinamento/Validação (6.326 amostras) e 20% para Teste puro (1.582 amostras).  É imperativo destacar que essa divisão foi realizada de forma estritamente estratificada por classe, utilizando uma semente aleatória fixa (random seed $=42$).  A estratificação garantiu que a proporção original de cada classe fosse preservada em ambas as partições.  A distribuição quantitativa exata dos dados particionados é detalhada na Tabela 11. 

Conforme evidenciado na Tabela 11, o conjunto de dados reflete o desbalanceamento natural intrínseco à produção agrícola, caracterizado por uma forte prevalência de qualidades superiores (ex: a classe majoritária 31.2 possui 1.407 amostras, enquanto a minoritária 44.4 possui apenas 215).  Para lidar com essa característica sem adulterar a distribuição estatística real do problema de negócio o que poderia gerar viés de amostragem, optou-se por não aplicar técnicas de superamostragem artificial no disco.  Em vez disso, a preservação da capacidade preditiva das classes minoritárias será tratada diretamente na fase de treinamento e avaliação dos modelos, por meio da adoção de métricas de otimização globalmente robustas a classes desbalanceadas, detalhadas na seção subsequente. 

---

### **PÁGINA 89**

3.5. Seleção e Adaptação dos Modelos Base (Modelos Avaliados) 
**Tabela 11 - Distribuição de amostras (patches) por classe e partição do dataset.** 

| Classe Comercial | Treino/Validação (80%) | Teste (20%) | Total |
| :--- | :---: | :---: | :---: |
| 31.2 | 1.126 | 281 | 1.407 |
| 31.3 | 975 | 244 | 1.219 |
| 31.4 | 1.112 | 278 | 1.390 |
| 32.3 | 858 | 215 | 1.073 |
| 32.4 | 275 | 69 | 344 |
| 33.3 | 1.034 | 259 | 1.293 |
| 41.4 | 238 | 59 | 297 |
| 41.5 | 234 | 58 | 292 |
| 42.4 | 302 | 76 | 378 |
| 44.4 | 172 | 43 | 215 |
| **Total Geral** | **6.326** | **1.582** | **7.908** |


87 
Nota: Divisão estratificada realizada antes da aplicação de técnicas de Data Augmentation. Fonte: Do autor. 

**3.5 Seleção e Adaptação dos Modelos Base (Modelos Avaliados)** 
Para determinar a arquitetura mais adequada ao desafio multiescala da classificação da pluma de algodão (que exige tanto a análise global da cor quanto a detecção local de micro-impurezas), estruturou-se um benchmark abrangente.  Em vez de focar em uma única topologia, a pesquisa avaliou um espectro de modelos representativos dos principais paradigmas evolutivos da Visão Computacional profunda.  As arquiteturas foram instanciadas através da biblioteca Torch Vision do PyTorch e agrupadas em três categorias de avaliação experimental: 

1.  **Modelos baseados em Autoatenção (Vision Transformers)**: Avaliou-se o ViT (Vision Transformer) em suas variantes ViT-B/16 e ViT-B/32, além do Swin Transformer hierárquico nas versões Tiny (Swin-T) e Small (Swin-S).  A inclusão destas redes visou testar a capacidade dos mecanismos de atenção global na captura da uniformidade estrutural da fibra. 
2.  **CNNs Modernas de Alta Eficiência**: Testou-se a família EfficientNet (variantes B0, B1 e B2) e a arquitetura ConvNeXt (variantes Tiny e Small). 

---

### **PÁGINA 90**

88 
Capítulo 3. Metodologia 

avaliar o balanço ótimo entre precisão de extração de características de granulação fina (fine-grained) e viabilidade computacional em cenários de inferência rápida. 
3.  **CNNs Clássicas e Robustas (Baselines)**: Foram incluídas arquiteturas que consagraram o paradigma do aprendizado profundo, englobando redes com roteamento residual (ResNet-18, ResNet-34 e ResNet-50), conectividade densa (DenseNet-121 e DenseNet-169), topologias multiescala (Inception-v3) e modelos estritamente sequenciais (VGG-16 e VGG-19, ambas em suas versões otimizadas com Batch Normalization). 

**3.5.1 Estratégia de Transfer Learning e Customização do Topo (Classification Head)** 
Devido à complexidade inerente de se treinar arquiteturas ultra-profundas (especialmente os Transformers) do zero com um conjunto de 7.908 tensores, adotou-se a estratégia de Transfer Learning (Transferência de Aprendizado).  Todos os backbones listados foram inicializados com pesos pré-treinados no dataset genérico ImageNet-1K (pesos padrão IMAGENET1K_V1).  Essa inicialização permitiu que as redes aproveitassem os filtros convolucionais primários responsáveis pela detecção de bordas, contrastes e texturas básicas, acelerando drasticamente a convergência e mitigando o risco de sobreajuste.  Apenas as camadas finais foram submetidas ao processo de "descongelamento" (unfreezing), cuja profundidade exata foi definida dinamicamente pelo algoritmo de otimização Optuna para cada rede, conforme documentado no Apêndice. 

Por fim, como as arquiteturas originais são projetadas para classificar 1.000 categorias genéricas, o componente de classificação original (Top Layer ou Classification Head) de cada rede foi removido.  Em sua substituição, acoplou-se um novo bloco classificador estritamente projetado para o escopo agrícola desta pesquisa.  A nova estrutura conectada à saída do backbone consistiu em: uma camada de redução de dimensionalidade por média global (Global Average Pooling);  uma camada de Dropout (com taxa estocástica também regulada pelo Optuna entre 0.0 e 0.7, dependendo da propensão da rede ao overfitting);  e uma camada densa e linear final configurada com exatamente 10 neurônios de saída, correspondentes às 10 classes comerciais de qualidade 

---

### **PÁGINA 91**

3.6. Ambiente de treinamento e avaliação dos modelos 
89 

da pluma de algodão avaliadas. A ativação preditiva final foi governada pela função Softmax durante o cálculo da perda de entropia cruzada. 

**3.6 Ambiente de treinamento e avaliação dos modelos** 
Para garantir a reprodutibilidade e a execução sistemática dos experimentos, desenvolveu-se um ambiente automatizado de treinamento.  A plataforma foi implementada na linguagem Python, utilizando o framework PyTorch devido à sua flexibilidade na manipulação de tensores e suporte nativo a grafos computacionais dinâmicos.  O processamento foi acelerado em hardware Apple Silicon (Mac M3), explorando o backend MPS (Metal Performance Shaders) para otimização da execução em GPU local.  A gestão do ciclo de vida dos experimentos foi conduzida através da plataforma MLflow, permitindo o rastreamento automatizado dos pesos das redes, hiperparâmetros testados e a evolução das métricas a cada época (epoch), assegurando a transparência e rastreabilidade total do processo. 

**3.6.1 Regularização e Aumento de Dados Dinâmico (On-the-fly)** 
Como o conjunto de dados original foi particionado sem a aplicação de superamostragem estática, a mitigação do viés de classes e a prevenção de overfitting foram tratadas de forma dinâmica durante o carregamento dos minilotes (batches).  Para isso, implementou-se um pipeline de aumento de dados em duas frentes distintas: transformações espaciais fotométricas e regularização a nível de lote. 

A primeira frente consistiu em transformações intra-imagem controladas pela biblioteca Albumentations (Buslaev et al., 2020).  A cada época de treinamento, as imagens do conjunto de treino foram submetidas estocasticamente a rotações afins (até 45°), espelhamentos horizontais e verticais, além de variações de brilho e contraste.  Essa abordagem garante a invariância rotacional do modelo, forçando-o a reconhecer a textura da pluma independentemente da sua orientação física na esteira. 

---

### **PÁGINA 92**

89 
Capítulo 3. Metodologia 

A segunda frente introduziu técnicas avançadas de regularização a nível de lote, especificamente o MixUp (Zhang et al., 2018) e o CutMix (Yun et al., 2019).  Ao contrário do aumento tradicional, o MixUp realiza uma interpolação linear entre os pixels e os rótulos de duas imagens aleatórias, enquanto o CutMix recorta um fragmento retangular de uma imagem e o sobrepõe em outra, misturando as proporções de suas classes.  A combinação dessas técnicas com o Label Smoothing (Szegedy et al., 2016) na função de perda (Cross-Entropy) provou-se fundamental para o domínio agrícola.  Elas penalizam a formação de predições superconfiantes (overconfident) e suavizam as fronteiras de decisão, incentivando a rede neural a focar em características estruturais da pluma em vez de memorizar ruídos ou o fundo das imagens. 

**3.6.2 Otimização Bayesiana com Optuna** 
As arquiteturas modernas de Deep Learning são altamente sensíveis à escolha de hiperparâmetros, cujo ajuste manual é ineficiente e propenso a subotimizações.  Para solucionar isso, a sintonia fina (fine-tuning) foi conduzida através de Otimização Bayesiana utilizando o framework Optuna.  Empregou-se o algoritmo TPE (Tree-structured Parzen Estimator), que modela a probabilidade condicional das métricas de desempenho dado um conjunto de hiperparâmetros, concentrando a busca em regiões promissoras do espaço paramétrico.  O espaço de busca foi customizado dinamicamente para cada família de arquiteturas, abrangendo restrições específicas para as peculiaridades de cada rede.  As variáveis otimizadas incluíram: 

* **Profundidade do Fine-Tuning**: O número de camadas descongeladas (unfreeze layers) para retreinamento, variando de acordo com a profundidade da rede (ex: poucas camadas para Vision Transformers para preservar a extração genérica, e camadas mais profundas para CNNs tradicionais). 
* **Taxa de Aprendizado (Learning Rate)**: Buscada em escala logarítmica (entre $10^{-5}e~10^{-2}$), acoplada a agendadores de decaimento (Schedulers) como Cosine 

---

### **PÁGINA 93**

3.6. Ambiente de treinamento e avaliação dos modelos 
Annealing Warm Restarts e ReduceLROnPlateau. 
91 

* **Decaimento de Peso (Weight Decay)**: Ajuste da regularização L2 para controle de complexidade dos pesos. 
* **Otimizadores**: Alternância adaptativa entre SGD (com momentum) para redes como ResNet, e AdamW, mandatório para a convergência de Transformers e ConvNeXt. 

**3.6.3 Função Objetivo Robusta para Desbalanceamento** 
Na otimização de hiperparâmetros, frequentemente nos deparamos com objetivos complementares: maximizar a acurácia de classificação e minimizar o erro de entropia cruzada.  Para consolidar essas metas em uma única métrica passível de otimização pelo algoritmo TPE, adotou-se a técnica de Escalarização Multiobjetivo (Knowles, 2006; Paria et al., 2020).  Desenvolveu-se uma Função Objetivo Composta que funde a correlação estatística inter-classes com a estabilidade do gradiente.  Minimizar diretamente a função de perda (Loss) ou maximizar a Acurácia Global em um dataset desbalanceado de pluma de algodão enviesaria o otimizador em favor dos hiperparâmetros que apenas classificam corretamente as classes majoritárias (como o Tipo 31.2).  Para garantir um aprendizado equitativo, desenvolveu-se uma Função Objetivo Composta, avaliada sobre o conjunto de validação, que funde a correlação estatística inter-classes com a estabilidade do gradiente.  A métrica base escolhida foi o Coeficiente de Correlação de Matthews (MCC), dada a sua robustez matemática inata perante o desequilíbrio de classes.  O valor escalar retornado ao Optuna a cada iteração (trial) foi calculado pela seguinte expressão: 
$O=\lambda_{1} \frac{(MCC+1)}{2} + \lambda_{2} (e^{-\theta_{val}})$ 
Onde a primeira parcela normaliza o MCC original (de [-1,1] para [0, 1]), ponderado pelo fator prioritário de classificação $\lambda_{1}=0.7$.  A segunda parcela introduz um 

---

### **PÁGINA 94**

92 
Capítulo 3. Metodologia 

termo de qualidade baseado na perda de validação $(.\theta_{val})$, aplicando uma decadência exponencial ponderada por $\lambda_{2}=0.3$.  Essa formulação garantiu que a busca Bayesiana priorizasse hiperparâmetros capazes de distinguir assertivamente todas as 10 classes de pluma, utilizando a suavidade da curva de perda apenas como critério de desempate técnico e estabilidade geométrica. 

**3.7 Desenho Experimental** 
Para consolidar o rigor científico desta pesquisa e permitir uma avaliação objetiva das arquiteturas selecionadas, formalizou-se um desenho experimental focado em responder às lacunas identificadas na literatura, especificamente quanto ao trade-off entre o poder de extração de características de granulação fina (fine-grained) e a eficiência computacional. 

**3.7.1 Hipóteses e Variáveis do Estudo** 
A experimentação foi conduzida para testar as seguintes hipóteses de pesquisa: 

* **Hipótese Nula $(H_{0})$**: Não existe diferença estatisticamente significativa no desempenho de classificação da qualidade da pluma de algodão entre arquiteturas baseadas em convolução (CNNs) e modelos baseados em autoatenção (Vision Transformers), quando avaliados sob as mesmas condições de iluminação e aumento de dados. 
* **Hipótese Alternativa $(H_{1})$**: Arquiteturas que combinam viés indutivo local com atenção global (como o Swin Transformer) ou CNNs de nova geração (ConvNeXt) apresentam um desempenho superior na distinção de classes visualmente similares, provendo um melhor equilíbrio entre acurácia e latência preditiva. 

Para testar estas hipóteses, o modelo experimental baseou-se nas seguintes variáveis: 

* **Variáveis Independentes**: A topologia da rede neural (arquiteturas listadas na seção anterior) e a configuração de hiperparâmetros (otimizada pelo Optuna). 

---

### **PÁGINA 95**

3.7. Desenho Experimental 
93 

* **Variáveis Dependentes**: O poder de generalização (mensurado por métricas de classificação sobre o conjunto de teste de 20%) e o custo computacional (mensurado pelo número de parâmetros, FLOPs e tempo de inferência por imagem). 

**3.7.2 Métricas de Avaliação** 
Devido ao inerente desbalanceamento de classes do domínio agroindustrial, a Acurácia Global (Overall Accuracy) foi monitorada apenas como métrica de referência estrutural, uma vez que ela pode apresentar um viés otimista ao mascarar o erro nas classes minoritárias.  A avaliação definitiva da capacidade de generalização dos modelos baseou-se nas seguintes métricas robustas: 

1.  **F1-Score Macro**: Avalia a média harmônica entre a Precisão e a Revocação (Recall).  O cálculo Macro computa a métrica separadamente para cada uma das 10 classes comerciais e extrai a média aritmética não ponderada.  Isso garante que as classes minoritárias (ex: Tipo 44.4) exerçam o mesmo peso na avaliação final que as majoritárias (ex: Tipo 31.2). 
2.  **Coeficiente de Correlação de Matthews (MCC)**: Considerada uma das métricas mais confiáveis para avaliação multiclasse em datasets severamente desequilibrados.  O MCC mede a correlação entre as predições e o ground truth, retornando um valor no intervalo de $[-1,+1]$, onde +1 representa uma predição perfeita, 0 equivale a uma predição aleatória $e-1$ denota discordância total. 
3.  **Coeficiente Kappa de Cohen (K)**: Estatística que mensura a concordância entre as predições do classificador e os rótulos reais, compensando a probabilidade de essa concordância ocorrer puramente ao acaso.  $\hat{E}$ uma métrica crucial para o domínio agrícola, pois penaliza severamente modelos enviesados que adotam a estratégia de prever apenas a classe dominante. 
4.  **Área Sob a Curva ROC Macro (Macro ROC AUC)**: Mensura a capacidade discriminativa do modelo na separação das classes de qualidade, independentemente do limiar de decisão (threshold) adotado pela função Softmax.  Para o problema multiclasse, o cálculo foi realizado utilizando a abordagem Um-Contra-Todos 

---

### **PÁGINA 96**

94 
Capítulo 3. Metodologia 

(One-vs-Rest- OvR) para cada tipo de algodão, extraindo-se em seguida a média não ponderada (Macro). 
5.  **Matriz de Confusão**: Utilizada para a análise qualitativa detalhada dos erros inter-classes.  Permite diagnosticar se os modelos estão confundindo amostras de algodão vizinhas no espectro de qualidade (ex: classificar o Tipo 31.2 como 31.3 devido a variações sutis de amarelamento). 
6.  **Eficiência Computacional**: Para chancelar a viabilidade da implementação da arquitetura vencedora em sistemas Edge Computing na indústria, aferiu-se o número total de parâmetros treináveis (em milhões) e as Operações de Ponto Flutuante por Segundo (FLOPs), correlacionando esse custo com o ganho preditivo. 

**3.7.3 Testes Estatísticos de Significância** 
A constatação de que um modelo "A" obteve um MCC fracionariamente maior que um modelo "B" no conjunto de teste não é suficiente para decretar sua superioridade na literatura científica.  Para garantir que as diferenças de desempenho observadas entre as melhores arquiteturas não decorreram de variações aleatórias da amostragem, aplicou-se o Teste Estatístico de McNemar.  O Teste de McNemar é uma prova não paramétrica aplicada a tabelas de contingência $2\times2$, amplamente recomendada na literatura de Machine Learning para comparar a proporção de erros e acertos de dois classificadores que foram avaliados sobre o exato mesmo conjunto de teste de hold-out (Dietterich, 1998).  Adotou-se um nível de significância $\alpha=0.05$ (Intervalo de Confiança de 95%).  Portanto, rejeitou-se a Hipótese Nula $(H_{0})$ apenas quando o valor-p (p-value) resultante da comparação arquitetural foi inferior a 0.05, atestando uma superioridade estatisticamente robusta na classificação da pluma de algodão. 

---

### **PÁGINA 97**

95 
CAPÍTULO 4 
RESULTADOS 

Neste capítulo, apresentamos os resultados obtidos nos experimentos de classificação automatizada de algodão, estruturados de forma a validar as hipóteses de pesquisa estabelecidas na metodologia.  A análise segue uma narrativa linear, partindo do estabelecimento de um baseline até a identificação do modelo com melhor equilíbrio entre performance e viabilidade industrial. 

**4.1 Evolução Experimental e Definição do Dataset Campeão** 
A pesquisa seguiu um processo evolutivo de refinamento dos dados e modelos.  Inicialmente, estabeleceu-se o baseline na versão V1 (RGB sob iluminação AB).  Em seguida, testou-se a incorporação de filtros clássicos na V8, e por fim, a otimização da iluminação física nas versões V9 e V10. 

A Figura 7 apresenta a distribuição do Matthews Correlation Coefficient (MCC) para todos os modelos ao longo dessas iterações.  Observa-se que a versão V10 elevou consistentemente o patamar de performance, superando significativamente as demais.  Para validar se essa diferença entre as versões é estatisticamente relevante, aplicou-se o teste de Friedman, complementado pelo teste post-hoc de Wilcoxon com 

---

### **PÁGINA 98**

96 
Capítulo 4. Resultados 

**Figura 7 - Evolução do MCC por versão do dataset. O salto de performance na V10 justifica sua seleção para as análises subsequentes.** 
(Gráfico de boxplot mostrando MCC de V1, V8, V9 e V10.) 

correção de Bonferroni. A Tabela 12 apresenta os postos médios e os grupos de significância. 

**Tabela 12 - Resultado detalhado do teste de Friedman e grupos de significância.** 

| Versão do Dataset | Posto Médio | Grupo |
| :--- | :---: | :---: |
| V10 | 1.25 | (a) |
| V9 | 2.12 | (a) |
| V1 | 2.62 | (a) |
| V8 | 4.00 | (c) |

Estatística de Friedman: $\chi^{2}=19.05$ $(p=2.67e-04)$
Grupos com mesma letra não apresentam diferença estatística $(p>0,05)$ 
Fonte: Do autor. 

A Figura 8 apresenta o diagrama de Diferença Crítica (CD).  O diagrama permite visualizar que a versão V10 (Posto Médio 1.12) e a versão V9 (Posto Médio 1.88) pertencem ao mesmo grupo de significância, sendo ambas estatisticamente superiores ao baseline (V1) e à versão experimental V8. 

---

### **PÁGINA 99**

**Figura 8 - Diagrama de Diferença Crítica (CD) entre as versões do dataset. Versões à esquerda apresentam melhor ranqueamento. A barra vermelha indica a distância mínima para que a diferença seja estatisticamente significativa.** 
(Diagrama linear comparando postos de V8, V1, V9 e V10.) 

global, a V10 obteve o maior MCC absoluto entre todos os experimentos.  Portanto, ela foi selecionada como o dataset padrão para as validações de hipóteses arquiteturais $(H_{1})$ e de eficiência industrial $(H_{4})$. 

**4.2 Validação da Hipótese $H_{1}$: Paradigma Arquitetural** 
Com o dataset V10 consolidado, avaliou-se a Hipótese $H_{1}$, que propõe a superioridade de arquiteturas de nova geração.  A Figura 9 confronta o desempenho das CNNs e dos Transformers.  Os resultados refutam a ideia de que Transformers puros (ViT) seriam superiores para este domínio.  A média de MCC das CNNs foi significativamente superior à dos Transformers.  Este comportamento sugere que para o reconhecimento de texturas da fibra de algodão, o viés indutivo de localidade das convoluções é mais eficaz que o mecanismo de atenção global, confirmando parcialmente $H_{1}$ apenas para a subfamília de CNNs modernas como a ConvNext. 

---

### **PÁGINA 100**

98 
Capítulo 4. Resultados 

**Figura 9 - Comparação de Paradigmas $(H_{1})$: CNNs vs. Transformers. Observa-se a liderança das CNNs densas e modernas.** 
(Gráfico de barras comparando INCEPTION, SWIN, VIT, RESNET, VGG, EFFICIENTNET, CONVNEXT, DENSENET.) 

**4.3 Impacto das Condições de Captura $(H_{2}e~H_{3})$** 
Nesta seção, isolamos as variáveis de iluminação física e pré-processamento digital para validar as hipóteses $H_{2}e~H_{3}$. 

**4.3.1 Impacto da Iluminação $(H_{2})$** 
A evolução do MCC através das diferentes fontes de luz (AB, AM, BR) é ilustrada na Figura 10. Observa-se um crescimento consistente na performance à medida que a iluminação é otimizada fisicamente, culminando na versão BR (V10).  Este resultado confirma $H_{2}$, demonstrando que a qualidade do sinal físico de entrada é um preditor de sucesso mais forte que a complexidade algorítmica isolada. 

**4.3.2 Expansão Espectral via Engenharia de Features $(H_{3})$** 
Em contraste, a tentativa de auxiliar os modelos via expansão espectral artesanal (15 canais Gabor) na versão V8 resultou em uma degradação severa, como mostrado na Figura 11. Este achado refuta $H_{3}$, indicando que para classificadores profundos, a inserção de features pré-computadas ruidosas pode atuar como um mecanismo de 

---

### **PÁGINA 101**

99 

**Figura 10 – Impacto da Iluminação (H2): Evolução do MCC para cada modelo conforme a fonte de luz é otimizada.** 

confusão no aprendizado end-to-end. 

**Figura 11 – Falha da Expansão Espectral (H3): O contraste entre a V1 (RGB) e a V8 (15-canais) mostra a queda drástica de performance.** 

---

### **PÁGINA 102**

100 Capítulo 4. Resultados

**4.4 Análise de Desempenho por Classe e Significância Estatística**
A análise detalhada da arquitetura campeã, a DenseNet V10, revela a robustez do modelo na distinção de classes comerciais de algodão.  A Tabela 13 apresenta as métricas por categoria no conjunto de teste. 

**Tabela 13 – Métricas detalhadas por classe para o modelo DenseNet V10.** 

| Classe | Precisão | Revocação | Escore F1 |
| :--- | :---: | :---: | :---: |
| 31.2 | 0.595 | 0.885 | 0.712 |
| 31.3 | 0.640 | 0.590 | 0.614 |
| 31.4 | 0.571 | 0.691 | 0.625 |
| 32.3 | 0.862 | 0.757 | 0.806 |
| 32.4 | 0.892 | 0.485 | 0.629 |
| 33.3 | 0.825 | 0.726 | 0.772 |
| 41.4 | 0.774 | 0.407 | 0.533 |
| 41.5 | 0.838 | 0.534 | 0.653 |
| 42.4 | 0.915 | 0.566 | 0.699 |
| 44.4 | 1.000 | 0.791 | 0.883 |

Fonte: Do autor. 

A Matriz de Confusão Normalizada (Figura 12) corrobora estes achados, mostrando que a maioria dos erros ocorre entre classes adjacentes no espectro morfológico.  Para aprofundar a análise, a Figura 13 apresenta a matriz de significância estatística (McNemar) entre os modelos.  Um ponto crucial reside na comparação entre a DenseNet (líder em MCC) e a ConvNext (segunda colocada).  O teste de McNemar revela que a superioridade da DenseNet sobre a ConvNext é estatisticamente significativa (p < 0,05).  Isso indica que a DenseNet possui uma capacidade superior de capturar padrões discriminativos sutis que a ConvNext, apesar de sua arquitetura moderna, não consegue replicar integralmente neste conjunto de dados. 

---

### **PÁGINA 103**

101 

**Figura 12 – Matriz de Confusão Normalizada para o modelo DenseNet V10.** 

**4.5 Eficiência e Viabilidade Industrial (H4)** 
A validação da Hipótese H4 exige o equilíbrio entre o rigor estatístico e a viabilidade operacional.  A Tabela 14 consolida os indicadores de produtividade para os modelos treinados na melhor versão do dataset (V10).  Para visualizar o compromisso entre as métricas, a Figura 14 apresenta o gráfico de dispersão correlacionando o MCC, a latência de inferência e a complexidade computacional (GFLOPs).  A análise da Figura 14 permite identificar três perfis tecnológicos distintos:

1.  **Alta Precisão (Elite)**: A DenseNet ocupa o topo do eixo Y, mas desloca-se para 

---

### **PÁGINA 104**

102 Capítulo 4. Resultados

**Figura 13 – Matriz de significância estatística (McNemar) no dataset V10. O sufixo (ns) indica ausência de diferença estatística.** 

**Tabela 14 – Ranking de Eficiência Industrial (Dataset V10).** 

| Model | MCC | Latency (ms) | GFLOPs | Throughput (img/s) |
| :--- | :---: | :---: | :---: | :---: |
| DENSENET | 0.646 | 19.440 | 2.865 | 51.441 |
| CONVNEXT | 0.615 | 6.823 | 4.470 | 146.559 |
| EFFICIENTNET | 0.591 | 13.559 | 0.681 | 73.754 |
| VGG | 0.491 | 11.648 | 19.551 | 85.851 |
| RESNET | 0.450 | 7.942 | 4.111 | 125.905 |
| VIT | 0.433 | 15.666 | 16.867 | 63.832 |
| SWIN | 0.414 | 11.884 | 4.509 | 84.144 |
| INCEPTION | 0.288 | 13.328 | 2.847 | 75.029 |

Fonte: Do autor. 

a direita no eixo de latência, indicando um custo computacional elevado para o ganho marginal de performance. 

---

### **PÁGINA 105**

103 

**Figura 14 – Trade-off Industrial: MCC vs. Latência (escala logarítmica). O tamanho das bolhas representa a complexidade em GFLOPs. Observa-se o agrupamento das CNNs modernas no quadrante superior esquerdo.** 

2.  **Ponto de Operação Industrial (Sweet Spot)**: A ConvNext situa-se no quadrante superior esquerdo, mantendo um MCC elevado com a menor latência entre os modelos de alta performance.  O tamanho moderado de sua bolha (GFLOPs) reforça sua adequação para sistemas de tempo real. 
3.  **Baixa Eficiência Relativa**: Os modelos baseados em Transformers (bolhas maiores e mais à direita) demonstram baixa eficiência para este problema específico, apresentando alta latência sem superar a barreira de performance das CNNs. 

Conclui-se que, embora a DenseNet possua superioridade estatística, a ConvNext emerge como a solução de maior viabilidade industrial, confirmando os pressupostos da hipótese H4 ao oferecer o melhor balanço entre agilidade e assertividade. 

---

### **PÁGINA 107**

105 Capítulo 5. Discussão 

CAPÍTULO 5
DISCUSSÃO

Neste capítulo, discutimos os achados experimentais sob a ótica dos paradigmas de arquitetura, o impacto das condições físicas e digitais de pré-processamento, e a viabilidade prática da implantação do sistema no ambiente industrial. 

**5.1 Análise dos Paradigmas de Arquitetura (CNN vs. Transformers)**
A comparação entre Redes Neurais Convolucionais (CNN) e modelos baseados em Atenção (Transformers) revelou uma superioridade consistente das arquiteturas convolucionais para o dataset de classificação de algodão.  Embora os modelos Vision Transformers (ViT) e Swin Transformer representem o estado da arte em tarefas de visão computacional generalista em larga escala, seu desempenho neste estudo foi limitado.  Este fenômeno pode ser atribuído à natureza do inductive bias (viés indutivo) inerente às CNNs.  As convoluções operam sob as premissas de localidade e invariância de translação, o que as torna extremamente eficientes na extração de características de textura e cor em conjuntos de dados de domínio específico e tamanho moderado.  Por outro lado, os Transformers possuem uma flexibilidade global que requer volumes massivos de dados para aprender relações espaciais que as CNNs já possuem estruturalmente.  No contexto da fibra de algodão, onde os padrões discriminativos são sutis e dependentes de 

---

### **PÁGINA 108**

106 Capítulo 5. Discussão

micro-texturas, o reuso de features da DenseNet e o refinamento do macro-design da ConvNext mostraram-se paradigmas mais robustos. 

**5.2 Impacto do Pré-processamento e Expansão Espectral**
O insucesso da versão experimental V8, que incorporou uma expansão para 15 canais utilizando filtros de Gabor e limiarização de Otsu, fornece um insight valioso sobre a engenharia de features em Deep Learning.  A queda drástica de desempenho em todos os modelos nesta versão evidencia o fenômeno da “Maldição da Dimensionalidade”.  Ao aumentar o espaço de entrada sem um aumento proporcional no volume de dados, elevou-se a complexidade do processo de otimização, inserindo possivelmente ruído artesanal que mascarou as representações latentes naturais dos pixels RGB.  Este resultado valida a tese de que, para classificadores profundos modernos, o aprendizado end-to-end (ponta a ponta) diretamente sobre os dados brutos tende a ser superior à tentativa de pré-induzir conhecimento através de filtros de visão computacional clássica, que podem não capturar a variabilidade estocástica inerente às fibras naturais de algodão. 

**5.3 A Influência da Iluminação na Generalização dos Modelos**
Um dos achados mais significativos deste estudo foi a variação de performance entre as versões V1 e V10, motivada primariamente pela alteração nas condições físicas de iluminação durante a captura.  A superioridade da V10 reforça o aforismo garbage in, garbage out (lixo entra, lixo sai) no contexto industrial.  Observou-se que a padronização e o espectro da fonte de luz têm um impacto mais direto na acurácia do que o ajuste fino de múltiplos hiperparâmetros.  Modelos treinados sob iluminação otimizada demonstraram menor entropia na função de perda e convergência mais estável.  Para o setor têxtil, este resultado sugere que o investimento em infraestrutura física de captura (câmaras de luz controladas) oferece um retorno sobre 

---

### **PÁGINA 109**

5.4. Análise de Erros e a Natureza Morfológica das Classes 107

investimento (ROI) em precisão algorítmica superior ao desenvolvimento de modelos excessivamente complexos que tentam compensar capturas ruidosas. 

**5.4 Análise de Erros e a Natureza Morfológica das Classes**
A análise qualitativa das classificações incorretas revelou que o sistema não falha de forma aleatória, mas sim em conformidade com as fronteiras morfológicas da fibra de algodão.  As confusões concentram-se predominantemente entre classes adjacentes (como Tipo 3 e Tipo 4), o que reflete a subjetividade intrínseca enfrentada por classificadores humanos.  O fato de a classe 41.4 ser frequentemente confundida com a 31.2 evidencia que existem limites físicos de discriminabilidade visual que até mesmo modelos de Deep Learning encontram dificuldade em superar.  Esta observação sugere que a classificação automatizada pode ser utilizada não apenas como um substituto, mas como uma ferramenta de apoio à decisão, fornecendo um grau de incerteza (probabilidade) que permite ao classificador humano intervir apenas nos casos de fronteira, reduzindo drasticamente o tempo total de processamento no emblocamento industrial. 

**5.5 Trade-off Industrial e Recomendações de Implantação**
A análise comparativa entre as arquiteturas revela uma dicotomia fundamental entre o rigor estatístico e a aplicabilidade em tempo real.  Embora o teste de McNemar tenha ratificado a superioridade da DenseNet-121 sobre a ConvNeXt-T com significância estatística (p < 0,05), a transposição desses resultados para o fluxo operacional da algodoeira exige uma interpretação pragmática das métricas.  A DenseNet, apesar de sua capacidade superior de preservação de microtexturas devido à conectividade densa, apresentou uma latência de inferência de 19,44 ms.  Em um cenário de automação industrial de alta velocidade, onde o sistema deve processar 

---

### **PÁGINA 110**

108 Capítulo 5. Discussão

múltiplos patches por fardo para compor uma nota global, esse tempo de resposta pode comprometer o throughput total da linha.  Por outro lado, a ConvNeXt emergiu como o ponto de operação ideal (sweet spot) para a indústria por três razões técnicas fundamentais: 

* **Eficiência de Processamento**: Com uma latência de apenas 6,82 ms, o modelo é aproximadamente 2,8 vezes mais rápido que a DenseNet, permitindo uma inspeção em tempo real sem gargalos logísticos.
* **Performance Competitiva**: Mesmo sendo a segunda colocada no ranking de MCC (0,615 vs 0,646), sua acurácia permanece em um patamar elevado e estável, superando significativamente os modelos baseados em atenção e as CNNs clássicas como ResNet e VGG. 
* **Modernização Arquitetural**: O uso de kernels de 7×7 e o design inspirado em Transformers conferem à ConvNeXt um campo receptivo ampliado, essencial para a análise morfológica das fibras, mantendo a simplicidade operacional das convoluções. 

Dessa forma, conclui-se que a ConvNeXt representa a solução de maior viabilidade para implementação em sistemas de Edge Computing.  A escolha prioriza a agilidade exigida pelo "emblocamento" industrial, aceitando um custo marginal de performance estatística em troca de uma robustez temporal inquestionável para o ambiente de produção. 

---

### **PÁGINA 111**

109 Capítulo 6. Conclusão

CAPÍTULO 6
CONCLUSÃO

Esta dissertação apresentou o desenvolvimento e a avaliação de um sistema de visão computacional voltado à classificação automatizada da pluma de algodão, comparando paradigmas de redes convolucionais e baseados em atenção.  A pesquisa demonstrou que a automação desse processo, historicamente subjetivo e logisticamente oneroso, é tecnicamente viável e estatisticamente robusta quando fundamentada em protocolos rígidos de aquisição física e modelos de aprendizado profundo de última geração. 

**6.1 Contribuições e Principais Achados**
As evidências experimentais permitiram validar as hipóteses propostas, resultando nas seguintes conclusões: 

* **Primazia da Iluminação (H2)**: A transição da versão V1 para a V10 do dataset confirmou que a otimização física da iluminação (uso de luz fria de 6000K para realce morfológico) é um fator de impacto superior ao ajuste fino de hiperparâmetros isoladamente.
* **Superioridade das CNNs sobre Transformers (H1)**: Contrariando tendências de domínios genéricos, as arquiteturas convolucionais superaram os Vision Transfor- 

---

### **PÁGINA 112**

110 Capítulo 6. Conclusão

mers (ViT). O viés indutivo de localidade das CNNs mostrou-se essencial para capturar as micro-texturas da fibra de algodão, especialmente em conjuntos de dados de domínio específico onde a convergência de mecanismos de atenção global é dificultada pela escassez de dados massivos. 
* **Inviabilidade da Engenharia de Features Manual (H3)**: A tentativa de expansão espectral via filtros de Gabor (V8) resultou em degradação de performance, ratificando que o aprendizado end-to-end sobre dados brutos é mais eficaz para modelar a variabilidade estocástica de fibras naturais. 
* **Eficiência Industrial (H4)**: Embora a DenseNet-121 tenha obtido a maior precisão estatística (MCC de 0,646), a ConvNeXt-T foi identificada como a solução de maior viabilidade prática.  Sua latência de 6,8 ms permite uma integração fluida em esteiras de alta velocidade, atendendo ao requisito de tempo real exigido pelo fluxo logístico das algodoeiras. 

**6.2 Limitações e Trabalhos Futuros**
Apesar do sucesso na classificação das 10 classes comerciais mais prevalentes, o estudo encontrou limites físicos de discriminabilidade visual em classes adjacentes.  Como desdobramentos desta pesquisa, sugerem-se: 

* **Integração de Metadados Físicos**: Combinar a análise visual com dados instrumentais do HVI (como Micronaire e Resistência) em uma arquitetura multimodal para reduzir a confusão entre tipos vizinhos.
* **Edge Computing e IoT**: Implementar o modelo ConvNeXt em dispositivos de borda integrados diretamente nos pontos de recepção de fardos e portos, permitindo uma auditoria de qualidade distribuída ao longo da cadeia logística. 
* **Detecção de Contaminantes Não-Botânicos**: Expandir o dataset para incluir materiais estranhos críticos (plásticos e óleos), explorando técnicas de segmentação de instância para quantificação precisa de impurezas. 
