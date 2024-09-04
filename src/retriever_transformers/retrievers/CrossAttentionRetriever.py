from typing import List
from transformers import AutoTokenizer, BatchEncoding
from torch.utils.data import DataLoader, Dataset
from torch.nn import Module
from dataclasses import dataclass
from typing import Callable

from tqdm import tqdm

import numpy as np

import torch

from ..models.CrossAttentionDistancePredictor import CrossAttentionDistancePredictor

class _BatchableDocuments(Dataset):
    def __init__(self, documents: BatchEncoding) -> None:
        self.documents = documents
    
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        return self.documents[idx]

@dataclass
class CrossAttentionRetrieverTrainingArguments():
    batch_size: int = 8
    shuffle: bool = False
    epochs: int = 1
    learning_rate: float = 1e-5
    step_callback: Callable[[float], None] = None
    freeze_llms: bool = False

@dataclass
class CrossAttentionRetrieverOutput():
    mrr: float
    accuracy: float

class _StandardMRRAndAccuracyEvaluatorDataset():
    def __init__(self, queries: List[str], documents: List[str]) -> None:
        self.queries = queries
        self.documents = documents
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        return self.queries[idx], self.documents[idx]

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
    def __init__(self, bert_checkpoint, seed=None, device=None):
        self.bert_checkpoint = bert_checkpoint
        self.model = CrossAttentionDistancePredictor(bert_checkpoint, seed=seed)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)
        self.device = device
        self.model.to(device)
    
    def _init_dataloader(self, queries: List[str], documents: List[str], batch_size: int = 8, shuffle: bool = False):
        dataset = _CrossAttentionRetrieverDataset(queries, documents)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _encode(self, texts: List[str]):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        if self.device is not None:
            inputs.to(self.device)
        return inputs
    
    def rank(self, queries: List[str], documents: List[str], batch_size: int = 16, progress_bar=False) -> CrossAttentionRetrieverOutput:
        self.model.eval()
        ranks = []
        if progress_bar:
            queries = tqdm(queries, desc="Ranking")
        for query in queries:
            ranks.append([])
            query = self._encode(query)
            documents_datasets = _BatchableDocuments(documents)
            documents_dataloader = DataLoader(documents_datasets, batch_size=min(batch_size, len(documents_datasets)))
            for batch in documents_dataloader:
                documents_tokenized = self._encode(batch)
                distances = self.model(query, documents_tokenized, rank_for_one_query=True)
                ranks[-1].extend(distances.detach().cpu().numpy())            
        return ranks
    
    def compute_mrr_and_accuracy(self, ranks: List[List[float]]) -> CrossAttentionRetrieverOutput:
        mrr = []
        accuracy = []
        for i, ranking in enumerate(ranks):
            best = np.argmax(ranking)
            sorted_ranking = np.flip(np.argsort(ranking))
            if best == i:
                accuracy.append(1)
            else:
                accuracy.append(0)
            mrr.append(1 / (np.where(sorted_ranking == i)[0] + 1))
        mrr = np.mean(mrr)
        accuracy = np.mean(accuracy)
        return CrossAttentionRetrieverOutput(mrr, accuracy)
    
    def _epoch_fit(self, dataloader, optimizer, loss_fn, step_callback: Callable[[float], None] = None, progress_bar=False):
        step = 0
        if progress_bar:
            dataloader = tqdm(dataloader, desc=f"Step {step}/{len(dataloader)}")
        for queries, documents, labels in dataloader:
            optimizer.zero_grad()
            query_embeddings = self._encode(queries)
            document_embeddings = self._encode(documents)
            labels = labels.to(self.device)
            logits = self.model(query_embeddings, document_embeddings)
            loss = loss_fn(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            if step_callback is not None:
                step_callback(loss)
            step += 1

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=False))

    def fit(self, queries: List[str], documents: List[str], args: CrossAttentionRetrieverTrainingArguments, epoch_callback: Callable[[int, Module], None] = None, progress_bar=False) -> Module:
        self.model.train(True)
        if args.freeze_llms:
            for param in self.model.query_model.parameters():
                param.requires_grad = False
            for param in self.model.answer_model.parameters():
                param.requires_grad = False
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.BCELoss().to(self.device)

        dataloader = self._init_dataloader(queries, documents, batch_size=args.batch_size * 2, shuffle=args.shuffle)
        epoch = 0
        if progress_bar:
            dataloader = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for epoch in range(args.epochs):
            self._epoch_fit(dataloader, optimizer, loss_fn, step_callback=args.step_callback, progress_bar=progress_bar)
            if epoch_callback is not None:
                epoch_callback(epoch, self.model)
        return self.model