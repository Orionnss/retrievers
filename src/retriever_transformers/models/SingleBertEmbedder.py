from torch.nn import Module
from transformers import BatchEncoding, AutoModel
from transformers.models.bert import BertModel
import torch

class SingleBertEmbedder(Module):
    def __init__(self, bert_checkpoint):
        super().__init__()
        self.model = AutoModel.from_pretrained(bert_checkpoint)

    def forward(self, batch: BatchEncoding):
        output = self.model(**batch, output_hidden_states=True, return_dict=True)
        embeddings = torch.mean(output[0], dim=1)
        return embeddings
    
