import torch
from torch.utils.data import DataLoader
from titans_model import TitansModel

# Загрузка тестовых данных
test_data = DataLoader(...)

# Инициализация модели
model = TitansModel(input_size=..., hidden_size=...)

# Оценка модели
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_data:
        inputs, targets = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = correct / total
print(f'Accuracy: {accuracy:.4f}')
