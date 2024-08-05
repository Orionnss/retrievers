from torch.nn import Module
from transformers import BatchEncoding, AutoModel
from transformers.models.bert import BertModel
import torch

class SingleBertEmbedder(Module):
    def __init__(self, bert_checkpoint):
        super().__init__()
        self.query_model = AutoModel.from_pretrained(bert_checkpoint)
        self.answer_model = AutoModel.from_pretrained(bert_checkpoint)

    def forward(self, query_batch: BatchEncoding, answer_batch: BatchEncoding):
        query_output = self.query_model(**query_batch, output_hidden_states=True, return_dict=True)
        answer_output = self.answer_model(**answer_batch, output_hidden_states=True, return_dict=True)
        query_embeddings = torch.mean(query_output[0], dim=1)
        answer_embeddings = torch.mean(answer_output[0], dim=1)
        return query_embeddings, answer_embeddings
    
