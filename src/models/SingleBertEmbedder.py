from torch.nn import Module
from transformers import AutoModelForCausalLM, BatchEncoding
import torch

class SingleBertEmbedder(Module):
    def __init__(self, bert_checkpoint):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(bert_checkpoint)

    def forward(self, input: BatchEncoding):
        output = self.model(**input)
        embeddings = torch.mean(output.last_hidden_state, dim=1)
        return embeddings
    
