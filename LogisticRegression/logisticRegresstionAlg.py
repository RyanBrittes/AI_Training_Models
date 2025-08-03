import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Criando um dataset fictício
data = {
    'x1': [2.5, 1.5, 3.2, 5.0, 6.3, 2.2, 3.3, 4.4, 4.8, 6.1],
    'x2': [2.4, 2.3, 3.0, 3.7, 4.4, 1.9, 3.0, 4.1, 4.3, 4.9],
    'y':  [0,   0,   0,   1,   1,   0,   0,   1,   1,   1]
}

df = pd.DataFrame(data)

# 2. Separar X e y
X = df[['x1', 'x2']].values
y = df['y'].values

# 3. Normalização (opcional, mas recomendado)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# 4. Inicialização dos parâmetros
m, n = X.shape
weights = np.zeros(n)
bias = 0
lr = 0.1
epochs = 1000

# 5. Função sigmoide
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 6. Treinamento com Gradiente Descendente
losses = []
for epoch in range(epochs):
    z = np.dot(X, weights) + bias
    y_pred = sigmoid(z)
    
    # Cálculo da perda (binary cross-entropy)
    loss = -np.mean(y * np.log(y_pred + 1e-15) + (1 - y) * np.log(1 - y_pred + 1e-15))
    losses.append(loss)

    # Gradientes
    dw = np.dot(X.T, (y_pred - y)) / m
    db = np.sum(y_pred - y) / m

    # Atualização dos parâmetros
    weights -= lr * dw
    bias -= lr * db

    # Print da perda a cada 100 épocas
    if epoch % 100 == 0:
        print(f'Época {epoch}, Loss: {loss:.4f}')

# 7. Predição
def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    probs = sigmoid(z)
    return (probs >= 0.5).astype(int)

# 8. Avaliação
y_pred = predict(X, weights, bias)
accuracy = np.mean(y_pred == y)
print(f"Acurácia final: {accuracy:.2f}")

# 9. Plotando a perda ao longo das épocas
plt.plot(losses)
plt.title('Loss durante o treinamento')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
