from torch.nn import Module, MultiheadAttention, Linear, Sigmoid, ReLU, TransformerDecoder, TransformerDecoderLayer
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
        self.cross_attention = TransformerDecoderLayer(768, 8, batch_first=True)
        self.linear = Linear(768, 1024)
        self.relu = ReLU()
        self.linear2 = Linear(1024, 1)
        self.sigmoid = Sigmoid()



    def forward(self, query_batch: BatchEncoding, answer_batch: BatchEncoding):
        query_output: BaseModelOutputWithPoolingAndCrossAttentions = self.query_model(**query_batch,
                                                                                      return_dict=True)
        answer_output: BaseModelOutputWithPoolingAndCrossAttentions = self.answer_model(**answer_batch,
                                                                                        return_dict=True)
        answer_embeddings = self.cross_attention(answer_output.last_hidden_state, query_output.last_hidden_state, 
                                                 tgt_is_causal=False,
                                                 memory_is_causal=False)
        final_token_embedding = answer_embeddings[:,0]
        out = self.linear(final_token_embedding)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        out = out.squeeze(-1)
        return out
