import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Gerando dados artificiais
np.random.seed(0)
n_amostras = 100
X = np.random.rand(n_amostras, 3)  # 3 features
true_weights = np.array([4.2, -3.5, 2.7])
bias = 5.0
noise = np.random.randn(n_amostras) * 0.2

# Saída real com ruído
y = X @ true_weights + bias + noise

print(y)

# Inicialização dos parâmetros
w = np.zeros(X.shape[1])  # vetor com 3 pesos (um para cada feature)
b = 0.0
lr = 0.1
epochs = 1000
n = len(y)
losses = []

# Função de custo
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Treinamento com Gradient Descent
for epoch in range(epochs):
    y_pred = X @ w + b
    error = y_pred - y

    # Derivadas parciais vetoriais
    dw = (2/n) * (X.T @ error)  # gradiente para w
    db = (2/n) * np.sum(error)  # gradiente para b

    # Atualização
    w -= lr * dw
    b -= lr * db

    loss = mse(y, y_pred)
    losses.append(loss)

    if (epoch + 1) % 100 == 0:
        print(f"Época {epoch+1}: Loss = {loss:.4f}, w1 = {w[0]:.4f}, b = {b:.4f}")

# Resultados
print("Pesos aprendidos:", w)
print("Bias aprendido:", b)

# Curva de erro
plt.plot(losses)
plt.xlabel("Época")
plt.ylabel("Erro (MSE)")
plt.title("Convergência da Regressão Linear Múltipla")
plt.grid()
plt.show()
