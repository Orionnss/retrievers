from .retriever import Retriever

class EmbedderRetriever(Retriever):
    def __init__(self, model):
        super().__init__()