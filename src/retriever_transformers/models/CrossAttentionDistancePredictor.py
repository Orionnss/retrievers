from torch.nn import Module, MultiheadAttention, Linear, Sigmoid
from transformers import BatchEncoding, AutoModel
from transformers.models.bert import BertModel
import torch

class CrossAttentionDistancePredictor(Module):
    def __init__(self, bert_checkpoint, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.query_model = AutoModel.from_pretrained(bert_checkpoint)
        self.answer_model = AutoModel.from_pretrained(bert_checkpoint)
        self.cross_attention = MultiheadAttention(embed_dim=768, num_heads=8)
        sequence_length = 512
        self.linear =  Linear(768*sequence_length, 1)
        self.sigmoid = Sigmoid()


    def forward(self, query_batch: BatchEncoding, answer_batch: BatchEncoding):
        query_output = self.query_model(**query_batch, output_hidden_states=True, return_dict=True)
        answer_output = self.answer_model(**answer_batch, output_hidden_states=True, return_dict=True)
        query_attn, _ = self.cross_attention(query_output[0], answer_output[0], answer_output[0])
        return self.sigmoid(self.linear(query_attn.flatten(start_dim=1))).view(query_batch["input_ids"].shape[0], -1)
    
