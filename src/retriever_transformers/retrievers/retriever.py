from typing import Any, List


class Retriever():
    def __init__(self):
        self.corpus = None

    def rank(self, queries: List[str], documents: List[str], progress_bar: bool = False):
        raise NotImplementedError
    
    def __call__(self, queries: List[str], documents: List[str]) -> Any:
        return self.rank(queries, documents, progress_bar=True)