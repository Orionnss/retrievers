from typing import Any


class Retriever():
    def __init__(self):
        self.corpus = None

    def encode(self, input):
        raise NotImplementedError
    
    @corpus.setter
    def corpus(self, corpus):
        raise NotImplementedError
    
    def query(self, queries):
        raise NotImplementedError
    
    def __call__(self, queries) -> Any:
        return self.query(queries)