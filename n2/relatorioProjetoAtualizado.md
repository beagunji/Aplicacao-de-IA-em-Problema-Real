==================================================
<br>
RELATÓRIO DO PROJETO
N2
<br>==================================================

Integrantes:

- Arthur de Pina Balduino Leitão - RA: 10400677 - 10400677@mackenzista.com.br

- Beatriz Hitomi Gunji - RA: 10402435 - 10402435@mackenzista.com.br

- Guilherme de Abreu Schulz - RA: 10401501 - 10401501@mackenzista.com.br

- Vinicius Moutinho Salvino - RA: 10402761 - 10402761@mackenzista.com.br

<br>

# Implementação de Machine Learning e Deep Learning sobre o gerenciamento de vendas de chocolates


## Resumo

O mercado de chocolates está em crescimento, impulsionado pelo aumento da renda, mudanças no consumo e investimentos comerciais. No Brasil, marcas como Cacau Show e Brasil Cacau expandiram significativamente suas operações. No entanto, desafios como previsão de demanda e gestão de estoques exigem soluções eficazes. A Inteligência Artificial (IA) se destaca como uma ferramenta importante estratégico para o negócio ao possibilitar análises preditivas de vendas e identificação de padrões de consumo, otimizando a gestão organizacional. O projeto propõe um framework baseado em Machine Learning e Deep Learning para prever demandas e tendências do mercado de chocolates, auxiliando na tomada de decisões. Os dados analisados incluem informações sobre vendedores, produtos, preços e quantidades vendidas, coletados de transações verificadas. A aplicação da IA permitirá às empresas a otimizar estoques, aprimorar estratégias de venda e compreender melhor os consumidores, garantindo maior eficiência operacional e vantagem competitiva. A metodologia começa com a análise exploratória dos dados, identificando os padrões e variáveis relevantes. Em seguida, realiza-se a preparação dos dados e serão desenvolvidos modelos para avaliar as métricas, assegurando confiabilidade. Espera-se obter um sistema preditivo capaz de auxiliar a gestão organizacional, prevendo demandas, otimizando operações, melhorando núcleo de vendas e fortalecendo a cultura organizacional orientada por dados, logo, promovendo uma maior eficiência e competitividade no mercado de chocolates.
<br>

## Introdução e Descrição do Problema

O mercado de chocolates é um segmento em crescimento do setor alimentício potencializado pelas tendências de consumo e sazonalidade com valor de vendas US$ 107 bilhões mundialmente em 2012. Envolve toda a cadeia, desde o cultivo do cacau, os processos de fabricação até comercialização do produto pronto (NETO; FRANCESCONI; PEDROSO, 2021). De acordo com Sebrae, o Brasil é o terceiro maior consumidor de chocolate no mundo e o nicho gourmet cresceu três vezes mais que o tradicional (DANTAS; PIRES; UETANABARO; GOMES; NOVAES, 2020). A tabela 1 abaixo, representa os principais tipos de chocolates existentes no mercado. 

<br>

| Tipo de chocolate | Definição |
|------------------|-----------|
| **Fino** | Produzido a partir de amêndoas de cacau com parâmetros organolépticos (aroma e sabor) diferenciados. |
| **Gourmet** | Produzido a partir de ingredientes de alta qualidade, por profissional especializado e com apresentação diferenciada. |
| **De origem** | Produzido a partir de amêndoa de cacau que possui nuances do local como aromas e sabores decorrentes do clima e da geografia local (terroir). |
| **Premium** | Produzido a partir de um tipo ou variedade do cacau de um local e de uma determinada safra. Muitas vezes é também classificado como fino ou gourmet. |

Tabela 1 – Tipos de chocolates 
<br>
Fonte: Dantas et al., 2020.
<br>

Segundo a Association of Chocolate, Biscuit and Confectionary Industries of Europe existem sete categorias de produtos de chocolate: sólido/tablete sem recheio (com adicional de nozes, cereais, frutas) ao leite ou puro; barra ou tablete recheado; bombom e outros confeccionados com álcool ou com/sem recheio; outros tipos de produtos que contenham caramelo ou revestidos com chocolate; chocolate branco; pasta de chocolate e em pó pronto para consumo (achocolatados e bebidas que contém cacau) (NETO; FRANCESCONI; PEDROSO, 2021). 

Os principais fatores que impulsiona este segmento no Brasil são:  crescimento da renda média da população, mudanças de hábito dos consumidores, investimentos de empresas em variadas áreas do mercado de chocolate, chegada de marcas internacionais e a busca/exploração de novos mercados por parte das empresas. Por exemplo a Cacau Show que ingressou em 1988 no segmento do chocolate, inaugurou em 2000 a 2013, 1500 lojas. Já Brasil Cacau, abriu 460 lojas (NETO; FRANCESCONI; PEDROSO, 2021). 

As empresas que atuam nesse mercado enfrentam desafios como previsão de demandas, otimização do processo de venda e do gerenciamento de estoques, e definição estratégias baseando-se no comportamento do consumidor.  

Neste contexto, a Inteligência Artificial (IA) surge como uma aliada poderosa, pois é um dos domínios mais fascinantes e com crescimento mais acelerado da atualidade, impulsionando inovações em diversas áreas do conhecimento. Não se limita apenas em compreender a inteligência, mas busca construir entidades capazes de tomar decisões eficazes e seguras em diversos contextos e situações (RUSSELL; NORVIG, 2022).  

Com a IA, houve avanços significativos na análise e previsão de vendas e demandas, auxiliando as organizações a tomarem decisões mais assertivas e eficientes para evitar o desperdício, reduzir os custos, mapear os consumidores e garantir que os produtos estejam disponíveis nos momentos certos, logo, sendo uma oportunidade estratégica para o setor de chocolates. Além disso, possibilitou a identificação de padrões de consumo e respectivos clientes com mais facilidade. Com isso, o projeto se justifica pelo potencial de aprimoramento da gestão comercial por meio da Inteligência Artificial. 

A abordagem escolhida para este projeto é o Framework na qual aplicará técnicas de Machine Learning e Deep Learning para desenvolver um modelo preditivo capaz de analisar padrões de venda de chocolates e prever demandas e tendências de mercado e, também envolverá uma análise exploratória dos dados. Os modelos buscarão identificar padrões de consumo, otimizar o funcionamento das vendas, oferecer insights para otimizar e direcionar as tomadas de decisão em relação a estoque, estratégias, gestão, contribuindo para o aumento da eficiência operacional e comercial das organizações deste setor. 

O objetivo deste projeto é aplicar os conceitos, modelos e técnicas de Inteligência Artificial para implementar soluções que visam melhorar a eficiência do setor de vendas de chocolates e permitir a análise de tendências com maior credibilidade. 

O conjunto de dados utilizado neste projeto são: o nome do vendedor, país, tipo do produto (sabor do chocolate), data de venda, preço do produto e quantidade de caixas enviadas. Todos foram coletados a partir de transações confirmadas realizadas em mercados online e varejistas de chocolate. Para garantir a precisão dos dados, apenas as transações verificadas foram incluídas. Os valores de receita consideram os preços finais dos chocolates após a aplicação dos descontos, caso existissem. 

A análise desses dados com a IA permitirá o desenvolvimento de modelos que auxiliem empresas de chocolate a otimizar estratégias operacionais e de venda; melhorar o gerenciamento de estoque; compreender melhor os hábitos de consumo dos clientes; prever tendências de mercado e demandas. Portanto, este projeto visa proporcionar um diferencial competitivo para o setor de chocolates, tornando-se mais eficientes e tomadas de decisão orientadas por dados confiáveis. 

<br>

## Aspectos Éticos do uso da IA e a sua Responsabilidade no desenvolvimento da solução

A aplicação da Inteligência Artificial na gestão comercial do setor de chocolates traz avanços significativos, como previsões mais precisas da demanda, otimização de estoques e formulação de estratégias baseadas em padrões de consumo. No entanto, é essencial que a implementação dessas tecnologias seja feita com responsabilidade, seguindo princípios éticos fundamentais, como transparência, justiça e segurança dos dados (LOPES et al., 2023). 

A privacidade e a proteção das informações também são questões essenciais. Os dados usados na modelagem incluem transações comerciais, identificação de vendedores e precificação, o que exige conformidade com leis como a Lei Geral de Proteção de Dados no Brasil e o Regulamento Geral sobre a Proteção de Dados (GDPR) na União Europeia. Para garantir a segurança dessas informações e evitar usos indevidos, métodos como anonimização e criptografia devem ser aplicados (SOUZA, 2023). 

Os algoritmos de Machine Learning e Deep Learning são treinados com dados históricos, podendo reproduzir desigualdades estruturais do mercado. Para evitar impactos negativos, é fundamental realizar auditorias regulares, garantir que os dados sejam diversos e representativos e criar mecanismos que reduzam a reprodução de padrões discriminatórios (LOPES et al., 2023). 

Além disso, a transparência dos modelos é um fator imprescindível. Empresas que adotam essa tecnologia devem garantir que os processos sejam fáceis de entender. Isso aumenta a confiança nas previsões e embasa decisões estratégicas, como precificação, distribuição e segmentação de mercado, reduzindo riscos causados pela falta de clareza nos sistemas automatizados (MIT TECHNOLOGY REVIEW, 2024). 

A automação está transformando o mercado de trabalho e trazendo desafios. Quando máquinas e inteligência artificial assumem tarefas antes feitas por humanos, algumas profissões podem ser impactadas. Por isso, é fundamental investir em treinamentos e na capacitação profissional para que todos possam se adaptar a essa nova realidade (SOUZA, 2023). 

Se bem aplicada, a IA não precisa substituir pessoas, mas sim atuar como uma ferramenta de apoio, tornando o trabalho mais eficiente e produtivo. Com planejamento adequado, é possível aproveitar os benefícios da tecnologia sem comprometer empregos. O segredo está no equilíbrio: usar a tecnologia como aliada e garantir que todos tenham espaço para crescer nesse novo cenário (MIT TECHNOLOGY REVIEW, 2024). 

Por fim, para que a IA seja utilizada de forma ética, é essencial um compromisso com a inovação responsável. As empresas do setor de chocolates devem estabelecer diretrizes claras para a gestão de dados, assegurar o uso correto da tecnologia e promover um ambiente de negócios que una eficiência e justiça. Assim, a IA pode se tornar um diferencial estratégico, impulsionando a competitividade sem comprometer valores éticos essenciais (LOPES et al., 2023). 

<br>

## Dataset

### Descrição Conteúdo do Dataset 

O dataset "chocolate_sales.csv" contém dados de vendas de produtos de chocolate, com a seguinte estrutura: 

- Sales Person: Nome do vendedor ou representante comercial responsável pela venda 

- Country: País onde a venda foi realizada, permitindo análises geográficas 

- Product: Tipo de produto de chocolate vendido (como barras, bombons, caixas etc.) 

- Date: Data em que a venda foi registrada 

- Amount: Valor monetário da venda 

- Boxes Shipped: Quantidade de caixas/unidades enviadas ao cliente 

O dataset fornece um registro abrangente de transações comerciais no setor de chocolates, abrangendo diferentes mercados geográficos, produtos, e períodos de tempo. Esses dados permitem análises de desempenho de vendas, comportamento sazonal, eficácia dos vendedores, popularidade de produtos e variações regionais de demanda. 

### Processo de Análise Exploratória 

A análise exploratória foi conduzida em várias etapas para extrair insights significativos: 

1. Análise Estrutural Básica: 

    a. Verificação das dimensões do dataset (número de registros e variáveis) 

    b. Identificação dos tipos de dados em cada coluna 

    c. Detecção de valores nulos e registros duplicados 

    <br>

2. Análise Estatística Descritiva: 

    a. Cálculo de métricas centrais (média, mediana) e de dispersão (desvio padrão) 

    b. Identificação de valores mínimos e máximos para entender os limites dos dados 

    c. Análise de quartis para compreender a distribuição das variáveis numéricas 

<br>

3. Análise Temporal: 

    a. Identificação do período coberto pelos dados (data mais antiga à mais recente) 

    b. Verificação da distribuição das vendas ao longo do tempo 

    c. Identificação de padrões sazonais ou tendências de crescimento/queda 

<br>

4. Análise Categórica: 

    a. Contagem de valores únicos em cada variável categórica 

    b. Análise de frequência das categorias mais comuns 

    c. Distribuição de vendas por país, produto e vendedor 

<br>

5. Análise de Relacionamentos: 

    a. Cálculo de correlações entre variáveis numéricas 

    b. Identificação de relações entre valor de venda e quantidade de caixas 

    c. Análise de desempenho por segmentos (vendedores, países, produtos) 

<br>

6. Detecção de Anomalias: 

    a. Identificação de outliers nas variáveis de vendas e quantidades 

    b. Análise do impacto dos outliers nas estatísticas gerais 

    c. Quantificação de possíveis valores extremos usando o método IQR 

<br>

7. Visualizações: 

    a. Histogramas para entender a distribuição das vendas 

    b. Gráficos de barras para comparação entre categorias 

    c. Séries temporais para análise de tendências 

    d. Mapas de calor para visualizar correlações 

<br>

### Preparação dos Dados em Python 

O processo de preparação dos dados foi estruturado para transformar o dataset bruto em um formato adequado para análises avançadas e modelagem: 

1. Tratamento de Valores Faltantes: 

    a. Identificação de colunas com valores nulos 

    b. Preenchimento de valores numéricos faltantes com a mediana 

    c. Preenchimento de valores categóricos faltantes com o modo (valor mais frequente) 

<br>

2. Transformação de Tipos de Dados: 

    a. Conversão de strings de data para o formato datetime 

    b. Criação de variáveis temporais derivadas (ano, mês, trimestre, dia da semana) 

    c. Identificação de fins de semana para análises de sazonalidade 

<br>

3. Tratamento de Outliers: 

    a. Detecção de valores extremos usando o método do Intervalo Interquartil (IQR) 

    b. Aplicação de winsorização para limitar valores extremos sem removê-los 

    c. Criação de colunas com valores tratados mantendo os originais 

<br>

4. Normalização e Padronização: 

    a. Normalização Min-Max para escalar variáveis numéricas no intervalo [0,1] 

    b. Tratamento de assimetrias em distribuições usando transformação logarítmica 

    c. Padronização de nomenclatura de colunas para consistência 

<br>

5. Engenharia de Features: 

    a. Criação de variáveis derivadas como valor por caixa 

    b. Categorização de valores contínuos em faixas (baixo, médio, alto, muito alto) 

    c. Extração de features temporais para capturar sazonalidade 

<br>

6. Codificação de Variáveis Categóricas: 

    a. Aplicação de one-hot encoding para variáveis categóricas 

    b. Criação de variáveis dummy para países, produtos e outras categorias 

    c. Preservação das variáveis originais junto com as codificadas 

<br>

7. Finalização do Dataset Preparado: 

    a. Organização das colunas em ordem lógica 

    b. Salvamento do dataset preparado em formato CSV 

    c. Documentação das transformações aplicadas 

<br>

O dataset resultante "chocolate_sales_prepared.csv" está enriquecido com features adicionais, valores tratados e formatos padronizados, tornando-o pronto para aplicação de técnicas avançadas de análise e modelagem preditiva. 

<br>


## Metodologia 

A metodologia adotada neste projeto segue um framework baseado em técnicas de Inteligência Artificial, com foco em aprendizado de máquina (Machine Learning) e aprendizado profundo (Deep Learning), aplicados à previsão de vendas no setor de chocolates. O objetivo é construir modelos capazes de reconhecer padrões em dados históricos de transações comerciais, auxiliando na tomada de decisões mais eficientes e fundamentadas. 

O processo inicia-se com uma análise exploratória dos dados disponíveis, visando compreender a estrutura e a distribuição das variáveis, identificar tendências sazonais, variações regionais de consumo e possíveis correlações entre variáveis como país, tipo de produto, quantidade vendida e valor arrecadado. Essa etapa segue princípios apresentados por Russell e Norvig (2022), ao destacar a importância da observação e da análise criteriosa como base para o desenvolvimento de sistemas inteligentes. 

Na fase de preparação dos dados, são realizadas tarefas fundamentais como tratamento de valores ausentes, conversão de formatos de data, categorização de variáveis e normalização de colunas numéricas. Também são criadas variáveis derivadas com o intuito de enriquecer a base de informação e melhorar o desempenho dos modelos preditivos, conforme orientações metodológicas comuns na literatura de ciência de dados (SOUZA, 2023). 

Com os dados estruturados, serão desenvolvidos modelos de previsão com base em técnicas de aprendizado supervisionado. A seleção dos modelos mais adequados será guiada por critérios de desempenho e capacidade de generalização. Segundo Dantas et al. (2020), compreender os padrões de consumo em mercados como o de chocolates — altamente influenciado por sazonalidade e preferências regionais — é fundamental para garantir uma operação comercial eficiente. 

A avaliação dos modelos será realizada com base em métricas amplamente utilizadas, garantindo confiabilidade na interpretação dos resultados. Além disso, serão elaboradas visualizações que representem os principais padrões e insights descobertos, permitindo que os dados se transformem em valor estratégico para as empresas do setor. 

Toda a solução será documentada e publicada em repositório digital aberto, promovendo transparência, reprodutibilidade e acesso à metodologia desenvolvida, como incentivado por boas práticas em projetos de IA (LOPES et al., 2023). 


## Resultados

O dataset utilizado apresentou uma estrutura adequada e um volume razoável de amostras para a Análise Exploratória dos Dados de Vendas de Chocolates. O período coberto pelas transações vai de 03/01/2022 a 31/08/2022, totalizando 240 dias, ou aproximadamente oito meses de vendas. Ao todo, o conjunto de dados contém 1.094 registros distribuídos em 6 colunas: 'Sales Person', 'Country', 'Product', 'Date', 'Amount' e 'Boxes Shipped'. Destaca-se a ausência de valores nulos e registros duplicados, na qual indica uma excelente qualidade estrutural dos dados sem a necessidade de tratamentos iniciais mais complexos.  A Tabela 2, a seguir, apresenta uma amostra com as cinco primeiras linhas do dataset, enquanto a Tabela 3 detalha os tipos de dados correspondentes a cada coluna. 


| #  | Sales Person    | Country   | Product             | Date       | Amount  | Boxes Shipped |
|----|----------------|-----------|---------------------|------------|---------|--------------|
| 0  | Jehu Rudeforth | UK        | Mint Chip Choco     | 04-Jan-22  | $5,320  | 180          |
| 1  | Van Tuxwell    | India     | 85% Dark Bars       | 01-Aug-22  | $7,896  | 94           |
| 2  | Gigi Bohling   | India     | Peanut Butter Cubes | 07-Jul-22  | $4,501  | 91           |
| 3  | Jan Morforth   | Australia | Peanut Butter Cubes | 27-Apr-22  | $12,726 | 342          |
| 4  | Jehu Rudeforth | UK        | Peanut Butter Cubes | 24-Feb-22  | $13,685 | 184          |

Tabela 2 - Primeiras 5 linhas do dataset 

Fonte: Elaborada pelos autores.


| Column        | Data Type |
|---------------|-----------|
| Sales Person  | object    |
| Country       | object    |
| Product       | object    |
| Date          | object    |
| Amount        | object    |
| Boxes Shipped | int64     |

Tabela 3 - Tipos de dados 

Fonte: Elaborada pelos autores.


A análise das categorias revela que a coluna Sales Person apresenta 25 valores únicos (Tabela 4), enquanto a coluna Country possui 6 países distintos (Tabela 5), a coluna Product contém 22 produtos diferentes (Tabela 6) e a coluna Amount apresenta 827 valores únicos (Tabela 7), demonstrando uma diversidade significativa nas transações comerciais registradas. Em relação à variável numérica Boxes Shipped, o total acumulado de caixas enviadas é de 177.007 unidades, com os seguintes indicadores estatísticos: 

- Média: 161,80; 

- Mediana: 135,0; 

- Moda: 24; 

- Valor mínimo: 1; 

- Valor máximo: 709. 


| Sales Person         | Frequency |
|---------------------|-----------|
| Kelci Walkden       | 54        |
| Brien Boise         | 53        |
| Van Tuxwell         | 51        |
| Beverie Moffet      | 50        |
| Oby Sorrel          | 49        |
| Dennison Crosswaite | 49        |
| Ches Bonnell        | 48        |
| Gigi Bohling        | 47        |
| Karlen McCaffrey    | 47        |
| Curtice Advani      | 46        |

Tabela 4 - Top 10 valores mais frequentes 

Fonte: Elaborada pelos autores. 

| Country     | Frequency |
|-------------|-----------|
| Australia   | 205       |
| India       | 184       |
| USA         | 179       |
| UK          | 178       |
| Canada      | 175       |
| New Zealand | 173       |

Tabela 5 - Paises aparecem no dataset 

Fonte: Elaborada pelos autores. 

| Product              | Frequency |
|----------------------|-----------|
| Eclairs              | 60        |
| 50% Dark Bites       | 60        |
| Smooth Sliky Salty   | 59        |
| White Choc           | 58        |
| Drinking Coco        | 56        |
| Spicy Special Slims  | 54        |
| Organic Choco Syrup  | 52        |
| 85% Dark Bars        | 50        |
| Fruit & Nut Bars     | 50        |
| After Nines          | 50        |

Tabela 6 - Top 10 valores mais frequentes de produtos

Fonte: Elaborada pelos autores. 

| Amount  | Frequency |
|---------|-----------|
| $2,317  | 5         |
| $2,303  | 4         |
| $4,361  | 4         |
| $3,472  | 4         |
| $3,577  | 4         |
| $7,714  | 4         |
| $5,691  | 4         |
| $6,454  | 4         |
| $3,374  | 3         |
| $2,282  | 3         |

Tabela 7 - Top 10 valores mais frequentes de preços 

Fonte: Elaborada pelos autores. 


A preparação dos dados consistiu em conversão de colunas de data, normalização de variáveis numéricas na coluna Boxes Shipped, aplicando one-hot encoding para variáveis categóricas na coluna Country. Foram criadas 7 novas features. 

Os resultados obtidos após o treinamento de Classificação utilizando o modelo Random Forest, com os valores de vendas de chocolates classificados em três categorias (grande, média e pequena), são apresentados na tabela 8 a seguir. O desempenho foi avaliado com base nos indicadores precision, recall, f1-score e support.

| Classe     | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| grande     | 0.00      | 0.00   | 0.00     | 36      |
| media      | 0.44      | 0.57   | 0.50     | 94      |
| pequena    | 0.38      | 0.40   | 0.39     | 89      |
|            |           |        |          |         |
| accuracy   |           |        | 0.41     | 219     |
| macro avg  | 0.27      | 0.33   | 0.30     | 219     |
| weighted avg | 0.34    | 0.41   | 0.37     | 219     |

Tabela 8 – Resultado da Classifficação

Fonte: Elaborada pelos autores. 

A acurácia geral do modelo foi de 41%, com uma média do f1-score de 0.37, indicando um desempenho geral insatisfatório. 

A classe "grande" apresentou o pior desempenho, com 0% de precisão e recall, ou seja, o modelo não foi capaz de identificar corretamente nenhum exemplo dessa categoria. Com isso, nota-se um forte desbalanceamento entre as classes, com pouca representatividade da categoria "grande", além de uma variação limitada de atributos e variáveis relevantes que permitam ao modelo aprender padrões distintos. 

A categoria "média" obteve desempenho médio com 44% de precisão e f1-score de 0.50, sendo a classe com melhor reconhecimento pelo modelo. Já a categoria "pequena" teve desempenho inferior com f1-score de 0.39, indicando que o modelo foi capaz de captar em partes os padrões dessa classe, mas apresenta dificuldades ao diferenciar com clareza os diferentes grupos. 

Portanto, os resultados demonstram que o modelo teve baixa capacidade de generalização devido ao desbalanceamento entre as classes e à limitação dos atributos e variáveis disponíveis. Logo, impactou diretamente a aprendizagem dos padrões e o desempenho preditivo, especialmente nas classes mais extremas. 


## Conclusão

Este trabalho teve o objetivo de aplicar modelos de Machine Learning e Deep Learning e frameworks da Inteligência Artificial com a proposta de realizar uma análise e previsão de vendas de chocolates, setor que atualmente está em expansão no mercado. 

Por meio da análise exploratória do dataset composto por dados e informações reais de vendas confirmadas como país, vendedor, tipo de produto, valor e quantidade vendida, foram realizados procedimentos para preparar os dados (engenharia de atributos, tratamento de outliers, normalização e codificação de variáveis). 

O modelo preditivo de Classificação escolhido foi Random Forest e as classes foram divididas em três classes alta, média e baixa. Porém, os resultados obtidos não atenderam às expectativas de desempenho e da proposta. A acurácia foi de 41% e a média do F1-score foi 0,37. A categoria “grande” apresentou 0% de precisão e recall, “média” teve desempenho regular com precisão de 44% e F1-score de 0,50, e a categoria “pequena” com F1-score de 0,39, também abaixo do ideal. Os fatores que dificultaram o desenvolvimento do modelo foram desbalanceamento das classes, baixa variação nos atributos e insuficiência de dados.  

Portanto, embora os resultados esperados não foram plenamente alcançados, a análise exploratória foi concluída com sucesso e gerou insights sobre o comportamento de vendas. Os testes realizados evidenciam como a IA pode potencializar a gestão comercial dos chocolates, oferecendo perspectivas e otimização das operações das organizações. 

 

## Endereço do vídeo no YouTube 

Link: https://youtu.be/85cLrdfZWfs?si=_BKMF8p4hp0HTWCH


## Referências Bibliográfica

DANTAS, Paulo César Cruz; PIRES, Mônica De Moura; UETANABARO, Ana Paula Trovatti; GOMES, Andrea da Silva; NOVAES, Ana Carolina Pereira. O mercado de chocolate no sul da Bahia: estrutura, produção e comercialização. DRd - Desenvolvimento Regional em debate, [S. l.], v. 10, p. 56–75, 2020. DOI: 10.24302/drd.v10i0.2373. Disponível em: https://www.periodicos.unc.br/index.php/drd/article/view/2373. Acesso em: 28 mar. 2025. 


LOPES, Nadja Velasco de; MACOHIN, Aline; BARBOSA, Inah Lúcia; FAGUNDES, Vladimir; GRAÇA, Fabiano. Ética na Inteligência Artificial. Serpro, 20 out. 2023. Disponível em: https://www.serpro.gov.br/menu/noticias/noticias-2023/etica-na-ia. Acesso em: 29 mar. 2025. 


MIT TECHNOLOGY REVIEW. Empresas devem definir regras e princípios éticos para o uso de IA. MIT Technology Review, 22 jul. 2024. Disponível em: https://mittechreview.com.br/etica-empresas-uso-ia/. Acesso em: 29 mar. 2025. 


NETO, P. F.; FRANCESCONI, M.; PEDROSO, M. C. Uma análise estratégica sobre o mercado brasileiro de chocolates / A strategic analysis about the Brazilian chocolate market. Brazilian Journal of Business, [S. l.], v. 3, n. 4, p. 3108–3127, 2021. DOI: 10.34140/bjbv3n4-023. Disponível em: https://ojs.brazilianjournals.com.br/ojs/index.php/BJB/article/view/34826. Acesso em: 28 mar. 2025. 


RUSSELL, Stuart J.; NORVIG, Peter. Inteligência Artificial: Uma Abordagem Moderna. 4. ed. Rio de Janeiro: GEN LTC, 2022. E-book. p.8. ISBN 9788595159495. Disponível em: https://app.minhabiblioteca.com.br/reader/books/9788595159495/. Acesso em: 29 mar. 2025. 


SOUNDANKAR, Atharva. Chocolate Sales Dataset. Kaggle, 2025. Disponível em: https://www.kaggle.com/datasets/atharvasoundankar/chocolate-sales/data. Acesso em: 29 mar. 2025. 


SOUZA, Fernanda de. Ética e Inteligência Artificial: qual a relação e os desafios? Alura, 18 set. 2023. Disponível em: https://www.alura.com.br/artigos/etica-e-inteligencia-artificial?srsltid=AfmBOorm7HZCBL-CeYPV_vkadirbl1n2GfOu4frrhXgyYATzCDILhZjH. Acesso em: 29 mar. 2025. 

 
