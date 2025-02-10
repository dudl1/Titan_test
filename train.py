import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from model import Titans
from data_loader import get_data_loader
from config import Config

# Load config
config = Config()

# Load data
train_loader = get_data_loader(config, mode='train')

# Initialize model
model = Titans(config)
optimizer = AdamW(model.parameters(), lr=config.learning_rate)

# Training loop
for epoch in range(config.num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        output = model(batch['input'])
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(output, batch['target'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # Save model checkpoint
    torch.save(model.state_dict(), f"model_epoch_{epoch}.pt")
