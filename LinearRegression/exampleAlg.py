import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Criando um conjunto de dados simples com pandas
data = {
    'tamanho_m2': [50, 60, 70, 80, 90, 100],
    'preco_milhares': [150, 180, 210, 240, 270, 300]  # Preço em milhares de reais
}
df = pd.DataFrame(data)

# 2. Separando entrada (X) e saída (y)
X = df[['tamanho_m2']].values.astype(np.float32)  # formato (N, 1)
y = df[['preco_milhares']].values.astype(np.float32)  # formato (N, 1)

# 3. Convertendo para tensores do PyTorch
X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# 4. Definindo o modelo de regressão linear
class RegressaoLinear(nn.Module):
    def __init__(self):
        super(RegressaoLinear, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

modelo = RegressaoLinear()

# 5. Função de perda e otimizador
criterio = nn.MSELoss()  # Mean Squared Error
otimizador = torch.optim.SGD(modelo.parameters(), lr=0.001)

# 6. Treinamento do modelo
epochs = 1000
for epoca in range(epochs):
    # Forward pass: predição
    y_pred = modelo(X_tensor)

    # Calculando o erro
    perda = criterio(y_pred, y_tensor)

    # Zerando gradientes anteriores
    otimizador.zero_grad()

    # Backward pass: calculando gradientes
    perda.backward()

    # Atualizando os pesos
    otimizador.step()

    # Exibindo a perda a cada 100 épocas
    if (epoca+1) % 100 == 0:
        print(f'Época [{epoca+1}/{epochs}], Perda: {perda.item():.4f}')

# 7. Exibindo os pesos treinados
for nome, parametro in modelo.named_parameters():
    print(f'{nome}: {parametro.data.numpy()}')

# 8. Fazendo predições com novos dados
tamanho_novo = torch.tensor([[110.0]])
preco_previsto = modelo(tamanho_novo)
print(f"Preço previsto para casa de 110m²: {preco_previsto.item():.2f} mil reais")

# 9. (Opcional) Visualização
X_numpy = X_tensor.numpy()
y_numpy = y_tensor.numpy()
y_pred_plot = modelo(X_tensor).detach().numpy()

plt.scatter(X_numpy, y_numpy, label='Dados reais')
plt.plot(X_numpy, y_pred_plot, color='red', label='Regressão Linear')
plt.xlabel('Tamanho (m²)')
plt.ylabel('Preço (milhares de R$)')
plt.legend()
plt.title('Regressão Linear com PyTorch')
plt.grid(True)
plt.show()
