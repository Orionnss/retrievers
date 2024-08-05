from ...src.retriever_transformers.models.SingleBertEmbedder import SingleBertEmbedder
from transformers import AutoTokenizer

model = SingleBertEmbedder("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def test_SimpleBertEmbedding():
    inputs = tokenizer(["Hello, my dog is cute"], return_tensors="pt")
    outputs = model(inputs)
    assert outputs is not None
    assert outputs.shape == (1, 768)

def test_SimpleBertEmbedding_batch():
    inputs = tokenizer(["Hello, my dog is cute", "Hello, my cat is cute"], return_tensors="pt")
    outputs = model(inputs)
    assert outputs is not None
    assert outputs.shape == (2, 768)
