import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error

# --- 1. Carregar e Pré-processar os Dados ---

# Carregar o dataset
df = pd.read_csv('auto-mpg.csv')

print("Primeiras 5 linhas do DataFrame:")
print(df.head())

# Substituir '?' por NaN e converter para numérico
df = df.replace('?', np.nan)
df['horsepower'] = pd.to_numeric(df['horsepower'])

# Remover linhas com valores ausentes
df.dropna(inplace=True)

# Selecionar apenas as features numéricas para simplificar
features = ['mpg', 'cylinders', 'displacement', 'weight', 'acceleration', 'model year']
X = df[features]
y = df['horsepower']

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Pré-processamento Concluído ---")
print("Formato dos dados de treino:", X_train.shape)
print("Formato dos dados de teste:", X_test.shape)

# --- 2. Treinar o Multilayer Perceptron (MLP) ---

# Criar e treinar o modelo MLP
mlp = MLPRegressor(hidden_layer_sizes=(100, 50),  # Duas camadas ocultas
                   activation='relu',              # Função de ativação ReLU
                   solver='adam',                  # Otimizador Adam
                   max_iter=2000,                  # Aumentar iterações para convergência
                   random_state=42)

print("\n--- Iniciando treinamento do MLP ---")
mlp.fit(X_train_scaled, y_train)
print("--- Treinamento concluído ---")

# --- 3. Avaliar o Modelo ---

print("\n--- Avaliação do Modelo ---")

# Fazer previsões
y_pred_train = mlp.predict(X_train_scaled)
y_pred_test = mlp.predict(X_test_scaled)

# Calcular métricas
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print(f"R² no conjunto de TREINO: {r2_train:.4f}")
print(f"R² no conjunto de TESTE: {r2_test:.4f}")
print(f"MSE no conjunto de TREINO: {mse_train:.4f}")
print(f"MSE no conjunto de TESTE: {mse_test:.4f}")

# Análise adicional
diferenca_r2 = r2_train - r2_test
print(f"\nDiferença R² (Treino - Teste): {diferenca_r2:.4f}")
if diferenca_r2 < 0.05:
    print("✅ Modelo com baixo overfitting!")
else:
    print("⚠️ Possível overfitting detectado.")

print(f"Modelo explica {r2_test*100:.1f}% da variância nos dados de teste.")

print("\n--- Processo Concluído ---")