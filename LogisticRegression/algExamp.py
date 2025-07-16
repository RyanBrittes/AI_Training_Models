import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. Gerando os dados simulados
np.random.seed(42)
num_samples = 200

humidity = np.random.uniform(30, 100, num_samples)
temperature = np.random.uniform(10, 35, num_samples)
wind_speed = np.random.uniform(0, 40, num_samples)

# Simulando probabilidade de chuva
rain_prob = (
    0.04 * humidity
    - 0.03 * temperature
    + 0.02 * wind_speed
    + np.random.normal(0, 1, num_samples)
)
labels = (rain_prob > 2.0).astype(int)

# Criando DataFrame
df = pd.DataFrame({
    'humidity': humidity,
    'temperature': temperature,
    'wind_speed': wind_speed,
    'rain': labels
})

# 2. Preparando os dados
X = df[['humidity', 'temperature', 'wind_speed']].values
y = df['rain'].values.reshape(-1, 1)

# Normalização
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_norm = (X - X_mean) / X_std

# Tensores
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 3. Modelo de Regressão Logística
class RainPredictionModel(nn.Module):
    def __init__(self):
        super(RainPredictionModel, self).__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 4. Treinamento
model = RainPredictionModel()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

losses = []
for epoch in range(500):
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Época {epoch+1}/500 - Loss: {loss.item():.4f}')

# 5. Visualizando a perda
plt.plot(losses)
plt.title("Função de perda")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# 6. Avaliação do modelo
with torch.no_grad():
    predictions = model(X_tensor)
    predicted_classes = (predictions >= 0.5).float()
    accuracy = (predicted_classes == y_tensor).sum().item() / y_tensor.size(0)
    print(f"Acurácia final: {accuracy * 100:.2f}%")

# 7. Salvando o modelo
MODEL_PATH = "/workspaces/AI_Training_Models/LogisticRegression/rain_model.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'mean': X_mean,
    'std': X_std
}, MODEL_PATH)
print(f"Modelo salvo em {MODEL_PATH}")

# 8. Carregando o modelo salvo
loaded_model = RainPredictionModel()
checkpoint = torch.load(MODEL_PATH, weights_only=False)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

# 9. Usando o modelo carregado para prever
# Exemplo: umidade = 85%, temperatura = 18°C, vento = 10 km/h
sample = np.array([[50, 30, 10]])
sample_norm = (sample - checkpoint['mean']) / checkpoint['std']
sample_tensor = torch.tensor(sample_norm, dtype=torch.float32)

with torch.no_grad():
    prediction = loaded_model(sample_tensor).item()
    print(f"Probabilidade de chover: {prediction:.2f}")
