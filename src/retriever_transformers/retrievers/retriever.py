from typing import Any, List, Callable
from dataclasses import dataclass
from torch import Tensor
from torch.nn import Module

@dataclass
class RetrieverTrainingArguments():
    batch_size: int = 8
    shuffle: bool = False
    epochs: int = 1
    learning_rate: float = 1e-5
    step_callback: Callable[[float], None] = None
    freeze_llms: bool = False

class RetrieverRankingOutput():
    ranks: Tensor
    

class Retriever():
    def __init__(self):
        self.corpus = None

    def rank(self, queries: List[str], documents: List[str], progress_bar: bool = False):
        raise NotImplementedError
    
    def __call__(self, queries: List[str], documents: List[str]) -> Any:
        return self.rank(queries, documents, progress_bar=True)
    
class TrainableRetriever(Retriever):
    def fit(self, queries: List[str], documents: List[str], args: RetrieverTrainingArguments, epoch_callback: Callable[[int, Module], None] = None):
        raise NotImplementedError