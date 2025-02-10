from transformers import LlamaTokenizer
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {'input': tokens['input_ids'].squeeze(), 'target': tokens['input_ids'].squeeze()}

def get_data_loader(config, mode='train'):
    tokenizer = LlamaTokenizer.from_pretrained(config.tokenizer_name)
    texts = [...]  # Load your dataset here
    dataset = TextDataset(texts, tokenizer, config.max_length)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=(mode == 'train'))
