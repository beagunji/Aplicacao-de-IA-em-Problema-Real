import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Verificar se o arquivo existe
file_path = 'chocolate_sales.csv'
if not os.path.exists(file_path):
    print(f"ERRO: O arquivo {file_path} não foi encontrado no diretório atual.")
    print(f"Diretório atual: {os.getcwd()}")
    print("Por favor, verifique o caminho do arquivo e tente novamente.")
else:
    print(f"Arquivo {file_path} encontrado! Iniciando a análise...")

    # Carregando o dataset
    try:
        df = pd.read_csv(file_path)
        print("\nDataset carregado com sucesso!")
        
        # 1. ANÁLISE EXPLORATÓRIA DOS DADOS
        print("\n" + "="*50)
        print("ANÁLISE EXPLORATÓRIA DOS DADOS DE VENDAS DE CHOCOLATES")
        print("="*50)
        
        # 1.1 Primeiras linhas do dataset
        print("\nPrimeiras 5 linhas do dataset:")
        print(df.head())
        
        # 1.2 Informações gerais
        print("\nInformações gerais do dataset:")
        print(f"Número de registros: {df.shape[0]}")
        print(f"Número de colunas: {df.shape[1]}")
        print("\nNomes das colunas:")
        print(df.columns.tolist())
        
        # 1.3 Tipos de dados
        print("\nTipos de dados:")
        print(df.dtypes)
        
        # 1.4 Estatísticas descritivas
        print("\nEstatísticas descritivas para dados numéricos:")
        print(df.describe())
        
        # 1.5 Verificando valores nulos
        null_counts = df.isnull().sum()
        print("\nQuantidade de valores nulos por coluna:")
        print(null_counts)
        print(f"\nPercentual de valores nulos no dataset: {(null_counts.sum() / (df.shape[0] * df.shape[1])) * 100:.2f}%")
        
        # 1.6 Verificando valores duplicados
        duplicates = df.duplicated().sum()
        print(f"\nNúmero de registros duplicados: {duplicates}")
        print(f"Percentual de duplicatas: {(duplicates / df.shape[0]) * 100:.2f}%")
        
        # 1.7 Análise da coluna de data (se existir)
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'data' in col.lower()]
        if date_columns:
            print("\nAnálise da(s) coluna(s) de data:")
            for date_col in date_columns:
                # Verificamos o formato atual antes de tentar converter
                print(f"\nColuna: {date_col}")
                print(f"Primeiros valores: {df[date_col].head().tolist()}")
                
                try:
                    df[date_col] = pd.to_datetime(df[date_col])
                    print(f"Data mais antiga: {df[date_col].min()}")
                    print(f"Data mais recente: {df[date_col].max()}")
                    print(f"Período de tempo: {(df[date_col].max() - df[date_col].min()).days} dias")
                except Exception as e:
                    print(f"Não foi possível converter para datetime. Erro: {e}")
        
        # 1.8 Análise das colunas categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            print("\nAnálise das colunas categóricas:")
            for col in categorical_cols:
                if col not in date_columns:
                    unique_values = df[col].nunique()
                    print(f"\nColuna: {col}")
                    print(f"Número de valores únicos: {unique_values}")
                    if unique_values <= 10:  # Se houver poucos valores únicos, exibimos todos
                        value_counts = df[col].value_counts()
                        print("Distribuição de valores:")
                        print(value_counts)
                    else:
                        top_values = df[col].value_counts().head(10)
                        print("Top 10 valores mais frequentes:")
                        print(top_values)
        
        # 1.9 Análise das colunas numéricas (se existirem amount, boxes, etc)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            # Análise de vendas
            amount_cols = [col for col in numeric_cols if 'amount' in col.lower() or 'valor' in col.lower() or 'venda' in col.lower() or 'sale' in col.lower()]
            if amount_cols:
                print("\nAnálise das colunas de valor/vendas:")
                for col in amount_cols:
                    print(f"\nColuna: {col}")
                    print(f"Total: {df[col].sum():,.2f}")
                    print(f"Média: {df[col].mean():,.2f}")
                    print(f"Mediana: {df[col].median():,.2f}")
                    print(f"Desvio padrão: {df[col].std():,.2f}")
                    print(f"Mínimo: {df[col].min():,.2f}")
                    print(f"Máximo: {df[col].max():,.2f}")
                    
                    # Verificando outliers com o método IQR
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                    print(f"Número de possíveis outliers: {outliers} ({outliers/len(df) * 100:.2f}%)")
            
            # Análise de caixas ou quantidades
            box_cols = [col for col in numeric_cols if 'box' in col.lower() or 'qtd' in col.lower() or 'quant' in col.lower() or 'shipped' in col.lower()]
            if box_cols:
                print("\nAnálise das colunas de caixas/quantidades:")
                for col in box_cols:
                    print(f"\nColuna: {col}")
                    print(f"Total: {df[col].sum():,}")
                    print(f"Média: {df[col].mean():,.2f}")
                    print(f"Mediana: {df[col].median():,}")
                    print(f"Moda: {df[col].mode()[0]:,}")
                    print(f"Mínimo: {df[col].min():,}")
                    print(f"Máximo: {df[col].max():,}")
        
        # 1.10 Análise de correlação entre variáveis numéricas
        if len(numeric_cols) >= 2:
            print("\nAnálise de correlação entre variáveis numéricas:")
            correlation = df[numeric_cols].corr()
            print(correlation)
            
            # Identificando as correlações mais fortes (positivas e negativas)
            corr_unstack = correlation.unstack()
            corr_filtered = corr_unstack[corr_unstack.abs() > 0.5]
            corr_filtered = corr_filtered[corr_filtered < 1.0]  # Removendo correlações de uma variável com ela mesma (=1)
            if not corr_filtered.empty:
                print("\nCorrelações mais fortes (|r| > 0.5):")
                for idx, corr_value in corr_filtered.items():
                    print(f"{idx[0]} e {idx[1]}: {corr_value:.2f}")
            else:
                print("\nNão foram encontradas correlações fortes (|r| > 0.5) entre as variáveis.")
        
        # 2. VISUALIZAÇÕES
        print("\n" + "="*50)
        print("VISUALIZAÇÕES DOS DADOS")
        print("="*50)
        
        # Configurando estilo das visualizações
        plt.style.use('ggplot')
        
        # 2.1 Distribuição de vendas (se houver coluna de valor)
        if amount_cols:
            main_amount_col = amount_cols[0]  # Usamos a primeira coluna de valor encontrada
            plt.figure(figsize=(10, 6))
            sns.histplot(df[main_amount_col], kde=True)
            plt.title('Distribuição do Valor de Vendas')
            plt.xlabel('Valor')
            plt.ylabel('Frequência')
            plt.savefig('chocolate_sales_distribution.png')
            print("\nGráfico de distribuição de vendas salvo como 'chocolate_sales_distribution.png'")
        
        # 2.2 Vendas por categoria (país, produto, etc.)
        if categorical_cols and amount_cols:
            # Tentamos encontrar colunas de país, produto ou vendedor
            category_cols = [col for col in categorical_cols if 'country' in col.lower() or 'pais' in col.lower() or 
                             'product' in col.lower() or 'produto' in col.lower() or 
                             'person' in col.lower() or 'vendedor' in col.lower()]
            
            if category_cols:
                main_category_col = category_cols[0]  # Usamos a primeira coluna categórica relevante
                main_amount_col = amount_cols[0]  # Usamos a primeira coluna de valor
                
                # Agrupando dados
                grouped_data = df.groupby(main_category_col)[main_amount_col].sum().sort_values(ascending=False)
                
                # Se tivermos muitas categorias, limitamos a 10
                if len(grouped_data) > 10:
                    grouped_data = grouped_data.head(10)
                
                plt.figure(figsize=(12, 6))
                grouped_data.plot(kind='bar')
                plt.title(f'Total de Vendas por {main_category_col}')
                plt.xlabel(main_category_col)
                plt.ylabel('Valor Total de Vendas')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f'chocolate_sales_by_{main_category_col.lower()}.png')
                print(f"\nGráfico de vendas por {main_category_col} salvo como 'chocolate_sales_by_{main_category_col.lower()}.png'")
        
        # 2.3 Séries temporais (se houver coluna de data)
        if date_columns and amount_cols:
            main_date_col = date_columns[0]  # Usamos a primeira coluna de data
            main_amount_col = amount_cols[0]  # Usamos a primeira coluna de valor
            
            # Convertendo para datetime se ainda não foi feito
            if not pd.api.types.is_datetime64_any_dtype(df[main_date_col]):
                try:
                    df[main_date_col] = pd.to_datetime(df[main_date_col])
                except:
                    print(f"\nNão foi possível converter a coluna {main_date_col} para datetime. Pulando análise temporal.")
                    pass
            
            if pd.api.types.is_datetime64_any_dtype(df[main_date_col]):
                # Criando coluna de mês-ano para agrupamento
                df['month_year'] = df[main_date_col].dt.to_period('M').astype(str)
                
                # Agrupando por mês
                monthly_sales = df.groupby('month_year')[main_amount_col].sum().reset_index()
                
                # Convertendo para datetime para ordenação correta
                monthly_sales['month_year'] = pd.to_datetime(monthly_sales['month_year'])
                monthly_sales = monthly_sales.sort_values('month_year')
                
                plt.figure(figsize=(14, 6))
                plt.plot(monthly_sales['month_year'], monthly_sales[main_amount_col], marker='o')
                plt.title('Tendência de Vendas ao Longo do Tempo')
                plt.xlabel('Mês')
                plt.ylabel('Valor Total de Vendas')
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('chocolate_sales_trend.png')
                print("\nGráfico de tendência de vendas salvo como 'chocolate_sales_trend.png'")
        
        # 2.4 Correlação entre colunas (heatmap)
        if len(numeric_cols) >= 2:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Matriz de Correlação')
            plt.tight_layout()
            plt.savefig('chocolate_sales_correlation.png')
            print("\nMapa de calor de correlação salvo como 'chocolate_sales_correlation.png'")
        
        # 3. PREPARAÇÃO DOS DADOS
        print("\n" + "="*50)
        print("PREPARAÇÃO DOS DADOS")
        print("="*50)
        
        # Criando uma cópia para não modificar o dataframe original
        df_prep = df.copy()
        
        # 3.1 Tratamento de valores nulos
        if null_counts.sum() > 0:
            print("\nTratamento de valores nulos:")
            for col in df_prep.columns:
                null_count = df_prep[col].isnull().sum()
                if null_count > 0:
                    print(f"Coluna {col}: {null_count} valores nulos")
                    if df_prep[col].dtype in ['int64', 'float64']:
                        # Para colunas numéricas, preenchemos com a mediana
                        median_value = df_prep[col].median()
                        df_prep[col].fillna(median_value, inplace=True)
                        print(f"  - Preenchidos com a mediana: {median_value}")
                    else:
                        # Para colunas categóricas, preenchemos com o valor mais frequente
                        mode_value = df_prep[col].mode()[0]
                        df_prep[col].fillna(mode_value, inplace=True)
                        print(f"  - Preenchidos com o valor mais frequente: {mode_value}")
        else:
            print("\nNão há valores nulos para tratar.")
        
        # 3.2 Conversão de tipos de dados (principalmente datas)
        if date_columns:
            print("\nConversão de colunas de data:")
            for date_col in date_columns:
                try:
                    if not pd.api.types.is_datetime64_any_dtype(df_prep[date_col]):
                        df_prep[date_col] = pd.to_datetime(df_prep[date_col])
                        print(f"Coluna {date_col} convertida para datetime.")
                        
                        # Criando features de data
                        df_prep[f'{date_col}_year'] = df_prep[date_col].dt.year
                        df_prep[f'{date_col}_month'] = df_prep[date_col].dt.month
                        df_prep[f'{date_col}_quarter'] = df_prep[date_col].dt.quarter
                        df_prep[f'{date_col}_day'] = df_prep[date_col].dt.day
                        df_prep[f'{date_col}_dayofweek'] = df_prep[date_col].dt.dayofweek
                        df_prep[f'{date_col}_is_weekend'] = df_prep[date_col].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
                        
                        print(f"Features temporais criadas a partir da coluna {date_col}.")
                except Exception as e:
                    print(f"Não foi possível converter a coluna {date_col}. Erro: {e}")
        
        # 3.3 Tratamento de outliers nas colunas numéricas (principalmente de valor)
        if amount_cols:
            print("\nTratamento de outliers:")
            for col in amount_cols:
                Q1 = df_prep[col].quantile(0.25)
                Q3 = df_prep[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_count = ((df_prep[col] < lower_bound) | (df_prep[col] > upper_bound)).sum()
                
                if outliers_count > 0:
                    # Criamos uma versão winsorizada (com outliers limitados)
                    df_prep[f'{col}_winsorized'] = df_prep[col].clip(lower_bound, upper_bound)
                    print(f"Coluna {col}: {outliers_count} outliers tratados com winsorização.")
                    print(f"  - Limite inferior: {lower_bound:.2f}")
                    print(f"  - Limite superior: {upper_bound:.2f}")
                    print(f"  - Nova coluna criada: {col}_winsorized")
        
        # 3.4 Normalização/Padronização de colunas numéricas
        if numeric_cols:
            print("\nNormalização de variáveis numéricas:")
            for col in numeric_cols:
                # Normalização Min-Max (escala 0-1)
                col_min = df_prep[col].min()
                col_max = df_prep[col].max()
                
                # Evitando divisão por zero
                if col_max > col_min:
                    df_prep[f'{col}_normalized'] = (df_prep[col] - col_min) / (col_max - col_min)
                    print(f"Coluna {col} normalizada (0-1). Nova coluna: {col}_normalized")
        
        # 3.5 One-hot encoding para variáveis categóricas
        if categorical_cols:
            print("\nAplicando one-hot encoding para variáveis categóricas:")
            for col in categorical_cols:
                # Se não for uma coluna de data e tiver poucos valores únicos (menos de 15)
                if col not in date_columns and df_prep[col].nunique() < 15:
                    dummies = pd.get_dummies(df_prep[col], prefix=col, drop_first=False)
                    df_prep = pd.concat([df_prep, dummies], axis=1)
                    print(f"One-hot encoding aplicado na coluna {col}. {dummies.shape[1]} novas colunas criadas.")
        
        # 3.6 Agregações e features derivadas
        if amount_cols and box_cols:
            print("\nCriando features derivadas:")
            amount_col = amount_cols[0]
            box_col = box_cols[0]
            
            # Calculando valor por caixa/unidade
            df_prep['value_per_box'] = df_prep[amount_col] / df_prep[box_col]
            print("Feature 'value_per_box' criada: valor por caixa/unidade.")
            
            # Categorizando o valor de venda
            df_prep['amount_category'] = pd.qcut(df_prep[amount_col], 4, labels=['Baixo', 'Médio', 'Alto', 'Muito Alto'])
            print("Feature 'amount_category' criada: categorização do valor de venda em quartis.")
        
        # 3.7 Log-transformação para distribuições muito assimétricas
        if amount_cols:
            for col in amount_cols:
                # Verificando assimetria
                skewness = df_prep[col].skew()
                if skewness > 1:  # Assimetria positiva significativa
                    # Aplicamos transformação log
                    df_prep[f'{col}_log'] = np.log1p(df_prep[col])
                    print(f"Coluna {col} tem assimetria forte ({skewness:.2f}). Log-transformação aplicada: {col}_log")
        
        # 3.8 Resumo do dataframe preparado
        print("\nDimensões do dataframe preparado:")
        print(f"Registros: {df_prep.shape[0]}")
        print(f"Colunas: {df_prep.shape[1]}")
        print(f"Novas features criadas: {df_prep.shape[1] - df.shape[1]}")
        
        # 3.9 Salvando o dataframe preparado
        df_prep.to_csv('chocolate_sales_prepared.csv', index=False)
        print("\nDataframe preparado salvo como 'chocolate_sales_prepared.csv'")
        
        # Resumo final
        print("\n" + "="*50)
        print("RESUMO DA ANÁLISE")
        print("="*50)
        print(f"1. Dataset original: {df.shape[0]} registros e {df.shape[1]} colunas")
        print(f"2. Dataset preparado: {df_prep.shape[0]} registros e {df_prep.shape[1]} colunas")
        
        if categorical_cols:
            main_cat_cols = [col for col in categorical_cols if col not in date_columns]
            if main_cat_cols:
                print("\nPrincipais variáveis categóricas:")
                for col in main_cat_cols[:3]:  # Mostramos até 3 principais
                    print(f"- {col}: {df[col].nunique()} valores únicos")
        
        if amount_cols:
            print("\nPrincipais métricas de venda:")
            main_amount = amount_cols[0]
            print(f"- Total de vendas: {df[main_amount].sum():,.2f}")
            print(f"- Ticket médio: {df[main_amount].mean():,.2f}")
        
        if date_columns and pd.api.types.is_datetime64_any_dtype(df[date_columns[0]]):
            print("\nPeríodo de análise:")
            main_date = date_columns[0]
            print(f"- De {df[main_date].min().strftime('%d/%m/%Y')} a {df[main_date].max().strftime('%d/%m/%Y')}")
            print(f"- Duração: {(df[main_date].max() - df[main_date].min()).days} dias")
        
        print("\nArquivos gerados:")
        print("1. chocolate_sales_prepared.csv - Dataset preparado para modelagem")
        print("2. Gráficos salvos na pasta atual")

    except Exception as e:
        print(f"Erro ao processar o dataset: {e}")