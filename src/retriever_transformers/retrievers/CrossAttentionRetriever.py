from typing import List
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from dataclasses import dataclass
from typing import Callable

import torch

from ..models.CrossAttentionDistancePredictor import CrossAttentionDistancePredictor

@dataclass
class CrossAttentionRetrieverTrainingArguments():
    batch_size: int = 8
    shuffle: bool = False
    epochs: int = 1
    learning_rate: float = 1e-5
    step_callback: Callable[[float], None] = None

@dataclass
class CrossAttentionRetrieverOutput():
    mrr: float
    accuracy: float

class _CrossAttentionRetrieverDataset(Dataset):
    def __init__(self, queries: List[str], documents: List[str]):
        self.queries = queries
        self.documents = documents

    def __len__(self):
        return len(self.queries) * 2

    def __getitem__(self, idx):
        if idx % 2 == 0:
            return self.queries[idx // 2], self.documents[idx // 2], torch.tensor(1, dtype=torch.float)
        else:
            return self.queries[idx // 2], self.documents[(idx // 2 - 1) % len(self.documents)], torch.tensor(0, dtype=torch.float)

class CrossAttentionRetriever():
    def __init__(self, bert_checkpoint, seed=None):
        self.bert_checkpoint = bert_checkpoint
        self.model = CrossAttentionDistancePredictor(bert_checkpoint, seed=seed)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
    
    def _init_dataloader(self, queries: List[str], documents: List[str], batch_size: int = 8, shuffle: bool = False):
        dataset = _CrossAttentionRetrieverDataset(queries, documents)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _encode(self, texts: List[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        return inputs
    
    def _epoch_fit(self, dataloader, optimizer, loss_fn, step_callback: Callable[[float], None] = None):
        for queries, documents, labels in dataloader:
            query_embeddings = self._encode(queries)
            document_embeddings = self._encode(documents)
            optimizer.zero_grad()
            logits = self.model(query_embeddings, document_embeddings)
            loss = loss_fn(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            if step_callback is not None:
                step_callback(loss)

    def fit(self, queries: List[str], documents: List[str], args: CrossAttentionRetrieverTrainingArguments, epoch_callback: Callable[[int, Module], None] = None) -> Module:
        self.model.train(True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.BCELoss()
        dataloader = self._init_dataloader(queries, documents, batch_size=args.batch_size * 2, shuffle=args.shuffle)
        for epoch in range(args.epochs):
            self._epoch_fit(dataloader, optimizer, loss_fn, step_callback=args.step_callback)
            if epoch_callback is not None:
                epoch_callback(epoch, self.model)
        return self.model