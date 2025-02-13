import torch
import torch.nn as nn

from long_term_memory import LongTermMemory
from attention_module import AttentionModule

class TitansModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TitansModel, self).__init__()
        self.long_term_memory = LongTermMemory(input_size, hidden_size)
        self.attention_module = AttentionModule(input_size, hidden_size)

    def forward(self, x):
        memory = self.long_term_memory(x)
        output = self.attention_module(x, memory)
        return output
