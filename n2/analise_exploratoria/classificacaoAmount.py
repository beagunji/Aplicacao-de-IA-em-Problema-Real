import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Carregar os dados
df = pd.read_csv("n2/analise_exploratoria/chocolate_sales_prepared.csv")

# Limpar e converter 'Amount'
df['Amount'] = df['Amount'].replace(r'[\$,]', '', regex=True).astype(float)

# Processar 'Date'
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.weekday

# Criar classes para 'Amount'
def classify_amount(value):
    if value <= 4000:
        return 'pequena'
    elif value <= 10000:
        return 'media'
    else:
        return 'grande'

df['Amount_Class'] = df['Amount'].apply(classify_amount)

# Definir atributos e alvo
features = ['Country', 'Product', 'Sales Person', 'Boxes Shipped', 'Month', 'Weekday']
X = df[features]
y = df['Amount_Class']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pré-processamento de colunas categóricas
categorical_features = ['Country', 'Product', 'Sales Person']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Pipeline de classificação
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Prever e gerar relatório
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
