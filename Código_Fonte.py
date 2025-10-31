import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score # Importando r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

#--------------------- Processamento dos dados --------------------#
# importando os dados
dataset_treino = pd.read_csv("TabelaTreino.csv", sep=';')

# Definindo a função de conversão
def convert_to_float(value):
    if isinstance(value, str):  # Verifica se o valor é uma string
        try:
            return float(value.replace('.', '').replace(',', '.'))  # Ajuste na conversão
        except ValueError:
            return np.nan  # Retorna NaN se não puder converter
    return value  # Retorna o valor se já for float

coluna_alterada = ['Abertura', 'Max', 'Min', 'Fechamento', 'Fechamento_Ajustado']

# Aplicar a substituição nas colunas
dataset_treino[coluna_alterada] = dataset_treino[coluna_alterada].replace('.', ',', regex=True)
# Remove as vírgulas da coluna "Volume" e converte para tipo numérico
dataset_treino['Volume'] = dataset_treino['Volume'].str.replace(',', '').astype(float)

dataset_treino['Fechamento'] = dataset_treino['Fechamento'].apply(convert_to_float)
# Selecionando a coluna que vai ser analisada
set_treino = dataset_treino['Fechamento'].values.reshape(-1, 1)
print(dataset_treino.head())

# Verificar se há valores NaN e removê-los
dataset_treino = dataset_treino.dropna()

# Dimensionando
sc = MinMaxScaler(feature_range=(0, 1))
set_treino_scaled = sc.fit_transform(set_treino)

# Criando a Estrutura de dados
X_treino = []
y_treino = []

for i in range(60, len(set_treino_scaled)):
    X_treino.append(set_treino_scaled[i-60:i, 0])
    y_treino.append(set_treino_scaled[i, 0])

X_treino, y_treino = np.array(X_treino), np.array(y_treino)

# Adicionando uma nova dimensão para converter de 2D para 3D
X_treino = np.reshape(X_treino, (X_treino.shape[0], X_treino.shape[1], 1))

#--------------------- Construindo o modelo --------------------#
# Inicializando
regressor = Sequential()

# Adicionando o LSTM com a camada de regularização dropout
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_treino.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Adicionando a camada de saída
regressor.add(Dense(units=1))

#--------------------- Compilando o modelo --------------------#
regressor.compile(optimizer='adam', loss='mean_squared_error')

#--------------------- Treinando o modelo --------------------#
early_stopping = EarlyStopping(monitor='loss', patience=10)
regressor.fit(X_treino, y_treino, epochs=840, batch_size=36, callbacks=[early_stopping])

#--------------------- Testando o modelo --------------------#
dataset_teste = pd.read_csv("TabelaTeste3M.csv", sep=';')
valor_real = dataset_teste['Fechamento'].apply(convert_to_float).dropna().values

# Verificar se há valores NaN e removê-los
dataset_teste = dataset_teste.dropna()

# Concatenando os dados de treino e os últimos 60 dias para prever os dados de teste
inputs = dataset_treino['Fechamento'].values[-60:].tolist() + valor_real.tolist()  # Pega os últimos 60 dias do treino e todos os dias do teste
inputs = np.array(inputs).reshape(-1, 1)
inputs_scaled = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs_scaled)):
    X_test.append(inputs_scaled[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Fazendo a previsão
previsao_valor = regressor.predict(X_test)
previsao_valor = sc.inverse_transform(previsao_valor)

#--------------------- Métricas de Avaliação --------------------#
# Calculando o erro percentual médio absoluto (MAPE)
mape = np.mean(np.abs((valor_real - previsao_valor) / valor_real)) * 100

# Calculando a acurácia (100 - MAPE)
accuracy_percentage = 100 - mape

# Calculando o Coeficiente de Determinação (R-squared)
r_squared = r2_score(valor_real, previsao_valor)

#--------------------- Visualizando os resultados --------------------#
plt.plot(valor_real, color='blue', label="Preço Real Ação 2024/2")
plt.plot(previsao_valor, color='green', label="Previsão Ação 2024/2")
plt.title(f"Predição valor Ação 3M\nMAPE: {mape:.2f}% - R-squared: {r_squared:.2f}")
plt.xlabel("Dias")
plt.ylabel("Preço Ação 3M")
plt.legend()
plt.show()