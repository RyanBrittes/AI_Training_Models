import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Gerar dados simulados: salário vs experiência
np.random.seed(42)
n = 100
x = np.random.rand(n) * 10  # anos de experiência (0 a 10)
y = 2.5 * x + 5 + np.random.randn(n) * 2  # salário com ruído

# Organizar com pandas
df = pd.DataFrame({'experience': x, 'salary': y})

# 2. Inicializar parâmetros
w = 0.0  # peso
b = 0.0  # bias

lr = 0.01  # taxa de aprendizado
epochs = 1000

# 3. Função de custo (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 4. Treinamento com gradient descent
losses = []
for epoch in range(epochs):
    y_pred = w * x + b
    error = y_pred - y

    # Derivadas da função de custo em relação a w e b
    dw = (2/n) * np.dot(error, x)
    db = (2/n) * np.sum(error)

    # Atualização dos parâmetros
    w -= lr * dw
    b -= lr * db

    # Registrar perda
    loss = mse(y, y_pred)
    losses.append(loss)

    if (epoch + 1) % 100 == 0:
        print(f"Época {epoch+1}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

nova_experiencia = 2
salario_previsto = w * nova_experiencia + b
print(f"Para {nova_experiencia} anos de experiência, o salário previsto é: R${salario_previsto:.2f} mil")