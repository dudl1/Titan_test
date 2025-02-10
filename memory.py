import torch
import torch.nn as nn

class LongTermMemory(nn.Module):
    def __init__(self, config):
        super(LongTermMemory, self).__init__()
        self.config = config
        self.memory = nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))
        self.momentum = nn.Parameter(torch.zeros(config.hidden_size, config.hidden_size))
        self.forget_gate = nn.Parameter(torch.ones(1))

    def update_memory(self, x, y):
        # Compute surprise (gradient of loss w.r.t. memory)
        loss = torch.nn.functional.mse_loss(self.memory @ x, y)
        grad = torch.autograd.grad(loss, self.memory)[0]
        
        # Update memory with momentum and forget gate
        self.momentum = self.momentum * 0.9 + grad * 0.1
        self.memory = (1 - self.forget_gate) * self.memory + self.momentum

    def retrieve_memory(self, x):
        return self.memory @ x
