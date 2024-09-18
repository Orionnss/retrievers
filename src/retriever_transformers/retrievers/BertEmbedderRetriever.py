from typing import List
from collections.abc import Callable
from dataclasses import dataclass

from networkx.algorithms import similarity
from transformers import AutoTokenizer
from ..models.SingleBertEmbedder import SingleBertEmbedder
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import TripletMarginLoss, Module, CosineSimilarity
from tqdm import tqdm
import torch
from torch import Tensor

from .retriever import TrainableRetriever, RetrieverTrainingArguments

@dataclass
class EmbedderRetrieverOutput():
    mrr: float
    accuracy: float

@dataclass
class EmbedderRetrieverTrainingArguments(RetrieverTrainingArguments):
    batch_size: int = 8
    shuffle: bool = False
    epochs: int = 1
    step_callback: Callable[[float], None] = None
    learning_rate: float = 1e-5


class _EmbedderRetrieverDataset(Dataset):
    def __init__(self, queries: List[str], documents: List[str]):
        self.queries = queries
        self.documents = documents

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.documents[idx]
    
class BertEmbedderRetriever(TrainableRetriever):
    def __init__(self, bert_checkpoint):
        super().__init__()
        self.model = SingleBertEmbedder(bert_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)

    def _encode(self, texts: List[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        embeddings = self.model(inputs)
        return embeddings
    
    def _init_dataloader(self, queries: List[str], documents: List[str], batch_size: int = 8, shuffle: bool = False):
        dataset = _EmbedderRetrieverDataset(queries, documents)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _epoch_fit(self, dataloader, optimizer, loss_fn, step_callback: Callable[[float], None] = None):
        for queries, documents in dataloader:
            optimizer.zero_grad()
            query_embeddings = self._encode(queries)
            document_embeddings = self._encode(documents)
            anchors = query_embeddings
            positives = document_embeddings
            negatives = torch.stack([ document_embeddings[i-1] for i in range(len(document_embeddings)) ])
            loss = loss_fn(anchors, positives, negatives)
            if step_callback is not None:
                step_callback(loss)
            loss.backward()
            optimizer.step()
    
    def fit(self, queries: List[str], documents: List[str], args: EmbedderRetrieverTrainingArguments, epoch_callback: Callable[[int, Module], None] = None, margin: float = 0.01):
        dataloader = self._init_dataloader(queries, documents, args.batch_size, args.shuffle)
        optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
        loss_fn = TripletMarginLoss(margin=margin)
        for epoch in range(args.epochs):
            self._epoch_fit(dataloader, optimizer, loss_fn, step_callback=args.step_callback)
            if epoch_callback is not None:
                epoch_callback(epoch, self.model)

    def evaluate(self, queries: List[str], documents: List[str], batch_size:int = 8):
        dataloader = self._init_dataloader(queries=queries, documents=documents, shuffle=False, batch_size=batch_size)
        rr = []
        accuracies = []
        similarity = CosineSimilarity(dim=1)
        for queries, documents in dataloader:
            queries_embeddings = self._encode(queries)
            documents_embeddings = self._encode(documents)
            for i, query in enumerate(queries_embeddings):
                scores = similarity(query, documents_embeddings)
                sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                rank = (sorted_indices == i).nonzero().item()
                rr.append(1/(rank+1))
                accuracies.append(sorted_indices[0] == i)
        mrr = sum(rr) / len(rr)
        accuracy = sum(accuracies) / len(accuracies)
        return EmbedderRetrieverOutput(mrr=mrr, accuracy=accuracy)

    def rank(self, queries: List[str], documents: List[str], progress_bar: bool = False):
        self.model.eval()
        ranks = torch.empty((100, 100))
        similarity = CosineSimilarity(dim=1)
        if progress_bar:
            queries = tqdm(queries, desc="Ranking")
        queries_embeddings = self._encode(queries)
        documents_embeddings = self._encode(documents)
        for i, query in enumerate(queries_embeddings):
            scores = similarity(query, documents_embeddings)
            ranks[i] = scores.detach()
        return ranks