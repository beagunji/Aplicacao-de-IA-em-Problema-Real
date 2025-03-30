Segue o output da análise exploratória realizada:

```
==================================================
ANÁLISE EXPLORATÓRIA DOS DADOS DE VENDAS DE CHOCOLATES
==================================================

Primeiras 5 linhas do dataset:
     Sales Person    Country              Product       Date    Amount  Boxes Shipped
0  Jehu Rudeforth         UK      Mint Chip Choco  04-Jan-22   $5,320             180
1     Van Tuxwell      India        85% Dark Bars  01-Aug-22   $7,896              94
2    Gigi Bohling      India  Peanut Butter Cubes  07-Jul-22   $4,501              91
3    Jan Morforth  Australia  Peanut Butter Cubes  27-Apr-22  $12,726             342
4  Jehu Rudeforth         UK  Peanut Butter Cubes  24-Feb-22  $13,685             184

Informações gerais do dataset:
Número de registros: 1094
Número de colunas: 6

Nomes das colunas:
['Sales Person', 'Country', 'Product', 'Date', 'Amount', 'Boxes Shipped']

Tipos de dados:
Sales Person     object
Country          object
Product          object
Date             object
Amount           object
Boxes Shipped     int64
dtype: object

Estatísticas descritivas para dados numéricos:
       Boxes Shipped
count    1094.000000
mean      161.797989
std       121.544145
min         1.000000
25%        70.000000
50%       135.000000
75%       228.750000
max       709.000000

Quantidade de valores nulos por coluna:
Sales Person     0
Country          0
Product          0
Date             0
Amount           0
Boxes Shipped    0
dtype: int64

Percentual de valores nulos no dataset: 0.00%

Número de registros duplicados: 0
Percentual de duplicatas: 0.00%

Análise da(s) coluna(s) de data:

Coluna: Date
Primeiros valores: ['04-Jan-22', '01-Aug-22', '07-Jul-22', '27-Apr-22', '24-Feb-22']
/Users/adepina/Desktop/topico7/topico7.py:67: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  df[date_col] = pd.to_datetime(df[date_col])
Data mais antiga: 2022-01-03 00:00:00
Data mais recente: 2022-08-31 00:00:00
Período de tempo: 240 dias

Análise das colunas categóricas:

Coluna: Sales Person
Número de valores únicos: 25
Top 10 valores mais frequentes:
Sales Person
Kelci Walkden          54
Brien Boise            53
Van Tuxwell            51
Beverie Moffet         50
Dennison Crosswaite    49
Oby Sorrel             49
Ches Bonnell           48
Karlen McCaffrey       47
Gigi Bohling           47
Curtice Advani         46
Name: count, dtype: int64

Coluna: Country
Número de valores únicos: 6
Distribuição de valores:
Country
Australia      205
India          184
USA            179
UK             178
Canada         175
New Zealand    173
Name: count, dtype: int64

Coluna: Product
Número de valores únicos: 22
Top 10 valores mais frequentes:
Product
50% Dark Bites         60
Eclairs                60
Smooth Sliky Salty     59
White Choc             58
Drinking Coco          56
Spicy Special Slims    54
Organic Choco Syrup    52
After Nines            50
85% Dark Bars          50
Fruit & Nut Bars       50
Name: count, dtype: int64

Coluna: Amount
Número de valores únicos: 827
Top 10 valores mais frequentes:
Amount
$2,317     5
$2,303     4
$7,714     4
$4,361     4
$6,454     4
$3,577     4
$3,472     4
$5,691     4
$994       3
$8,001     3
Name: count, dtype: int64

Análise das colunas de caixas/quantidades:

Coluna: Boxes Shipped
Total: 177,007
Média: 161.80
Mediana: 135.0
Moda: 24
Mínimo: 1
Máximo: 709

==================================================
VISUALIZAÇÕES DOS DADOS
==================================================

==================================================
PREPARAÇÃO DOS DADOS
==================================================

Não há valores nulos para tratar.

Conversão de colunas de data:

Normalização de variáveis numéricas:
Coluna Boxes Shipped normalizada (0-1). Nova coluna: Boxes Shipped_normalized

Aplicando one-hot encoding para variáveis categóricas:
One-hot encoding aplicado na coluna Country. 6 novas colunas criadas.

Dimensões do dataframe preparado:
Registros: 1094
Colunas: 13
Novas features criadas: 7

Dataframe preparado salvo como 'chocolate_sales_prepared.csv'

==================================================
RESUMO DA ANÁLISE
==================================================
1. Dataset original: 1094 registros e 6 colunas
2. Dataset preparado: 1094 registros e 13 colunas

Principais variáveis categóricas:
- Sales Person: 25 valores únicos
- Country: 6 valores únicos
- Product: 22 valores únicos

Período de análise:
- De 03/01/2022 a 31/08/2022
- Duração: 240 dias
```
