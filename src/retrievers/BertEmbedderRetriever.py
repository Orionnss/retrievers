from typing import List
from dataclasses import dataclass

from transformers import AutoTokenizer
from models.SingleBertEmbedder import SingleBertEmbedder
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn import TripletMarginLoss, Module

@dataclass
class EmbedderRetrieverTrainingArguments():
    batch_size: int = 8
    shuffle: bool = False
    epochs: int = 1
    step_callback: callable[[float]] = None


class _EmbedderRetrieverDataset(Dataset):
    def __init__(self, queries: List[str], documents: List[str]):
        self.queries = queries
        self.documents = documents

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.documents[idx]
    
class BertEmbedderRetriever():
    def __init__(self, bert_checkpoint):
        super().__init__()
        self.model = SingleBertEmbedder(bert_checkpoint)
        self.tokenizer = AutoTokenizer(bert_checkpoint)

    def _encode(self, input: List[str]):
        inputs = self.tokenizer(input, return_tensors="pt", padding=True, truncation=True)
        embeddings = self.model(inputs)
        return embeddings
    
    def _init_dataloader(self, queries: List[str], documents: List[str], batch_size: int = 8, shuffle: bool = False):
        dataset = _EmbedderRetrieverDataset(queries, documents)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def _epoch_fit(self, dataloader, optimizer, loss_fn, step_callback: callable[[float]] = None):
        for batch in dataloader:
            queries, documents = batch
            query_embeddings = self._encode(queries)
            document_embeddings = self._encode(documents)
            anchors = query_embeddings
            positives = document_embeddings
            negatives = [ document_embeddings[i-1] for i in range(len(document_embeddings)) ]
            optimizer.zero_grad()
            loss = loss_fn(anchors, positives, negatives)
            if step_callback is not None:
                step_callback(loss)
            loss.backward()
            optimizer.step()
    
    def fit(self, queries: List[str], documents: List[str], args: EmbedderRetrieverTrainingArguments, epoch_callback: callable[[int, Module]] = None):
        dataloader = self._init_dataloader(queries, documents, args.batch_size, args.shuffle)
        optimizer = Adam(self.model.parameters())
        loss_fn = TripletMarginLoss()
        for epoch in range(args.epochs):
            self._epoch_fit(dataloader, optimizer, loss_fn, step_callback=args.step_callback)
            epoch_callback(epoch, self.model)
            