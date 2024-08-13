from torch.nn import Module, MultiheadAttention, Linear, Sigmoid, ReLU, TransformerDecoder, TransformerDecoderLayer
from transformers import BatchEncoding, AutoModel
from transformers.models.bert import BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from typing import Optional, Callable

from torch import Tensor
import torch


class TransformerDecoderLayerWithAttentionWeights(TransformerDecoderLayer):
    def set_callback_for_attention_weights(self, callback: Callable[[Tensor], None]):
        self.callback = callback
    def _mha_block(self, x: Tensor, mem: Tensor,
                attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        
        output, weights = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=True)
        if hasattr(self, 'callback'):
            self.callback(weights)
        x = output[0]
        return self.dropout2(x)


class CrossAttentionDistancePredictor(Module):
    def __init__(self, bert_checkpoint, seed=None, attention_weights_callback: Optional[Callable[[Tensor], None]] = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.query_model: BertModel = AutoModel.from_pretrained(bert_checkpoint)
        self.answer_model: BertModel = AutoModel.from_pretrained(bert_checkpoint)
        self.cross_attention = TransformerDecoderLayerWithAttentionWeights(768, 8, batch_first=True)
        if attention_weights_callback is not None:
            self.cross_attention.set_callback_for_attention_weights(attention_weights_callback)
        self.linear = Linear(768, 1024)
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
        answer_embeddings = self.cross_attention(answer_output.last_hidden_state, query_hidden_state, 
                                                 tgt_is_causal=False,
                                                 memory_is_causal=False)
        final_token_embedding = answer_embeddings[:,0]
        out = self.linear(final_token_embedding)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        out = out.squeeze(1)
        return out
