import torch
import torch.nn as nn

class LongTermMemory(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LongTermMemory, self).__init__()
        self.hidden_size = hidden_size
        self.memory = nn.Parameter(torch.randn(input_size, hidden_size))

    def forward(self, x):
        # Обновление памяти на основе входных данных
        self.memory.data = self.memory.data + x
        return self.memory
