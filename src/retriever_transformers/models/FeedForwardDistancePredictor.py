from torch.nn import Module, MultiheadAttention, Linear, Sigmoid, ReLU, TransformerDecoder, TransformerDecoderLayer
from transformers import BatchEncoding, AutoModel
from transformers.models.bert import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from typing import Optional, Callable

from torch import Tensor
import torch

class FeedForwardDistancePredictor(Module):
    def __init__(self, bert_checkpoint, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.query_model: BertModel = AutoModel.from_pretrained(bert_checkpoint)
        self.answer_model: BertModel = AutoModel.from_pretrained(bert_checkpoint)
        self.linear = Linear(768*2, 1024)
        self.relu = ReLU()
        self.linear2 = Linear(1024, 1)
        self.sigmoid = Sigmoid()

    def forward(self, query_batch: BatchEncoding, answer_batch: BatchEncoding, rank_for_one_query=False):
            
        query_output: BaseModelOutputWithPoolingAndCrossAttentions = self.query_model(**query_batch,
                                                                                      return_dict=True)
        answer_output: BaseModelOutputWithPoolingAndCrossAttentions = self.answer_model(**answer_batch,
                                                                                        return_dict=True)
        if rank_for_one_query:
            batch_size = answer_output.last_hidden_state.shape[0]
            query_hidden_state = query_output.last_hidden_state.squeeze().repeat(batch_size, 1, 1)
        else:
            query_hidden_state = query_output.last_hidden_state
        out = self.linear(torch.cat([query_hidden_state, answer_output.last_hidden_state], dim=2))
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        out = out.squeeze(1)
        return out
