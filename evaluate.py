from model import Titans
from data_loader import get_data_loader
from utils import compute_perplexity

# Load model
model = Titans(config)
model.load_state_dict(torch.load("model_epoch_9.pt"))
model.eval()

# Load test data
test_loader = get_data_loader(config, mode='test')

# Evaluate
total_perplexity = 0
for batch in test_loader:
    output = model(batch['input'])
    perplexity = compute_perplexity(output, batch['target'])
    total_perplexity += perplexity.item()

print(f"Test Perplexity: {total_perplexity / len(test_loader)}")
