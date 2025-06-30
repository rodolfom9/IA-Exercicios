import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. Carregar e Pré-processar os Dados ---

# Carregar o dataset
df = pd.read_csv('auto-mpg.csv')

print("Primeiras 5 linhas do DataFrame:")
print(df.head())
print("\nInformações do DataFrame:")
print(df.info())

# A coluna 'horsepower' é carregada como object devido a valores '?'
# Vamos substituir '?' por NaN e converter para numérico
df = df.replace('?', np.nan)
df['horsepower'] = pd.to_numeric(df['horsepower'])

# Lidar com valores ausentes: remover linhas com NaN (apenas em 'horsepower' neste caso)
df.dropna(inplace=True)

# Separar features (X) e target (y)
X = df.drop('horsepower', axis=1)
y = df['horsepower']

# Identificar colunas numéricas e categóricas
# 'car name' não é uma feature útil para o modelo e será descartada pelo 'remainder' padrão
categorical_features = ['origin']
numerical_features = ['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'model year']

# Pré-processamento:
# - Escalonamento de features numéricas (StandardScaler)
# - Codificação de features categóricas (OneHotEncoder)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='drop' # Descarta colunas não especificadas (como 'car name')
)

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Pré-processamento Concluído ---")
print("Formato dos dados de treino (X_train):", X_train.shape)
print("Formato dos dados de teste (X_test):", X_test.shape)
print("Formato do target de treino (y_train):", y_train.shape)
print("Formato do target de teste (y_test):", y_test.shape)

# --- 2. Implementar e Treinar o Multilayer Perceptron (MLP) com Scikit-learn ---

# Construir o pipeline para o MLP do Scikit-learn
mlp_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', MLPRegressor(hidden_layer_sizes=(100, 50), # Duas camadas ocultas com 100 e 50 neurônios
                               activation='relu',            # Função de ativação ReLU
                               solver='adam',                # Otimizador Adam
                               max_iter=1000,                # Número máximo de iterações
                               random_state=42,
                               early_stopping=True,          # Ativar early stopping
                               n_iter_no_change=10,          # Número de épocas sem melhoria para parar
                               verbose=True))                # Exibir progresso do treinamento
])

# Treinar o modelo MLP
print("\n--- Iniciando treinamento do MLPRegressor do Scikit-learn ---")
mlp_pipeline.fit(X_train, y_train)
print("--- Treinamento do MLPRegressor concluído ---")

# --- 3. Avaliar o Modelo MLP ---

# Avaliar o modelo nos conjuntos de treino e teste
print("\n--- Avaliação do Modelo MLPRegressor ---")

# Previsões nos dados de treino e teste
y_pred_mlp_train = mlp_pipeline.predict(X_train)
y_pred_mlp_test = mlp_pipeline.predict(X_test)

# Calcular R^2 para treino e teste
mlp_r2_train = r2_score(y_train, y_pred_mlp_train)
mlp_r2_test = r2_score(y_test, y_pred_mlp_test)

# Calcular MSE para treino e teste
mlp_mse_train = mean_squared_error(y_train, y_pred_mlp_train)
mlp_mse_test = mean_squared_error(y_test, y_pred_mlp_test)

print(f"Desempenho do MLPRegressor (R^2) no conjunto de TREINO: {mlp_r2_train:.4f}")
print(f"Desempenho do MLPRegressor (R^2) no conjunto de TESTE: {mlp_r2_test:.4f}")
print(f"Erro Quadrático Médio (MSE) do MLPRegressor no conjunto de TREINO: {mlp_mse_train:.4f}")
print(f"Erro Quadrático Médio (MSE) do MLPRegressor no conjunto de TESTE: {mlp_mse_test:.4f}")

print("\n--- Processo do MLPRegressor Concluído ---")