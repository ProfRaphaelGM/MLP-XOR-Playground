import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

print("Iniciando o script...")  # Confirma o início do script

# Solicita que o aluno insira seu número de RM (somente dígitos)
rm = int(input("Digite o seu número de RM (apenas números): "))

# Define a seed para garantir resultados reprodutíveis e únicos para cada aluno
np.random.seed(rm)
tf.random.set_seed(rm)

# Dados de entrada para o problema XOR
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Saídas esperadas para o XOR
y = np.array([[0],
              [1],
              [1],
              [0]])

# Define o learning rate desejado
learning_rate = 0.01

# Criação do modelo: 1 camada oculta com 2 neurônios utilizando sigmoid e camada de saída com sigmoid
model = Sequential([
    Dense(2, input_dim=2, activation='sigmoid'),
    Dense(1, activation='sigmoid')
])

# Configura o otimizador com o learning rate especificado
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compilação do modelo com o otimizador definido e a função de perda binary_crossentropy
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento do modelo com um número maior de épocas
print("Treinando o modelo...")
history = model.fit(X, y, epochs=1500, verbose=1)

# Avaliação do modelo
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nAvaliação do modelo:\nLoss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Gera previsões para cada entrada do XOR
print("\nPrevisões para as entradas do XOR:")
predictions = model.predict(X)
for inp, pred in zip(X, predictions):
    print(f"Entrada: {inp} => Saída prevista: {pred[0]:.4f}")


# Exibe os pesos e bias de cada camada
print("\nPesos e Bias treinados:")
for idx, layer in enumerate(model.layers):
    pesos, bias = layer.get_weights()
    print(f"\nCamada {idx+1}:")
    print("Pesos:")
    print(pesos)
    print("Bias:")
    print(bias)
