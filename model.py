import torch
import torch.nn as nn
from attention import Attention
from memory import LongTermMemory

class Titans(nn.Module):
    def __init__(self, config):
        super(Titans, self).__init__()
        self.config = config
        self.core = Attention(config)  # Short-term memory (attention)
        self.long_term_memory = LongTermMemory(config)  # Long-term memory
        self.persistent_memory = nn.Parameter(torch.randn(config.persistent_memory_size))  # Persistent memory

    def forward(self, x):
        # Retrieve long-term memory
        h = self.long_term_memory.retrieve_memory(x)
        
        # Combine persistent memory with input
        x_combined = torch.cat([self.persistent_memory.unsqueeze(0).expand(x.size(0), -1), h, x], dim=1)
        
        # Process through core (attention)
        y = self.core(x_combined)
        
        # Update long-term memory
        self.long_term_memory.update_memory(x, y)
        
        return y
