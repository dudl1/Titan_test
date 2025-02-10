class Config:
    def __init__(self):
        self.hidden_size = 512
        self.num_layers = 6
        self.persistent_memory_size = 128
        self.max_length = 4096  # Context window size
        self.batch_size = 32
        self.learning_rate = 4e-4
        self.num_epochs = 10
        self.tokenizer_name = "llama-tokenizer"
