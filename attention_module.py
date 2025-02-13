import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionModule, self).__init__()
        self.query = nn.Linear(input_size, hidden_size)
        self.key = nn.Linear(input_size, hidden_size)
        self.value = nn.Linear(input_size, hidden_size)

    def forward(self, x, memory):
        q = self.query(x)
        k = self.key(memory)
        v = self.value(memory)
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output
