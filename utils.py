import torch

def compute_perplexity(logits, targets):
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    return torch.exp(loss)

def log_metrics(metrics):
    for key, value in metrics.items():
        print(f"{key}: {value}")
