from torch.nn import Module, MultiheadAttention, Linear, Sigmoid, ReLU
from transformers import BatchEncoding, AutoModel
from transformers.models.bert import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
import torch

class CrossAttentionDistancePredictor(Module):
    def __init__(self, bert_checkpoint, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.query_model: BertModel = AutoModel.from_pretrained(bert_checkpoint)
        self.answer_model: BertModel = AutoModel.from_pretrained(bert_checkpoint)
        self.cross_attention = MultiheadAttention(embed_dim=768, num_heads=8)
        #sequence_length = 512
        self.linear =  Linear(768, 32)
        self.relu = ReLU()
        self.linear2 = Linear(32, 1)
        self.sigmoid = Sigmoid()


    def forward(self, query_batch: BatchEncoding, answer_batch: BatchEncoding):
        query_output: BaseModelOutputWithPoolingAndCrossAttentions = self.query_model(**query_batch, output_hidden_states=True, return_dict=True)
        answer_output: BaseModelOutputWithPoolingAndCrossAttentions = self.answer_model(**answer_batch, output_hidden_states=True, return_dict=True)
        print(query_output.last_hidden_state.shape)
        print(answer_output.last_hidden_state.shape)
        query_attn, _ = self.cross_attention(query_output.last_hidden_state, answer_output.last_hidden_state, answer_output.last_hidden_state)
        return self.sigmoid(self.linear2(self.relu(self.linear(query_attn[:,0])))).view(-1)
    
