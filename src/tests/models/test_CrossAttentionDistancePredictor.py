from ...retriever_transformers.models.CrossAttentionDistancePredictor import CrossAttentionDistancePredictor
from transformers import AutoTokenizer
from matplotlib import pyplot as plt
from os.path import exists
import torch

model = CrossAttentionDistancePredictor("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def test_SimpleDistancePrediction():
    inputs_questions = tokenizer(["Hello, my dog is cute"], return_tensors="pt", padding="max_length", max_length=512, )
    inputs_answers = tokenizer(["Hello, my cat is cute"], return_tensors="pt", padding="max_length", max_length=512)
    outputs = model(inputs_questions, inputs_answers)
    assert outputs is not None
    assert outputs.shape == torch.Size([1])
    assert outputs[0] > 0 and outputs[0] < 1

def test_SimpleDistancePrediction_multiple():
    inputs_questions = tokenizer(["Hello, my dog is cute", "Hello, my cat is cute"], return_tensors="pt", padding="max_length", max_length=512)
    inputs_answers = tokenizer(["Hello, my cat is cute", "Hello, my dog is cute"], return_tensors="pt", padding="max_length", max_length=512)
    outputs = model(inputs_questions, inputs_answers)
    assert outputs is not None
    assert outputs.shape == torch.Size([2])
    assert outputs[0] > 0 and outputs[0] < 1
    assert outputs[1] > 0 and outputs[1] < 1
    assert outputs[0] != outputs[1]

def test_rank_for_one_query():
    inputs_questions = tokenizer(["Hello, my dog is cute"], return_tensors="pt", padding="max_length", max_length=512)
    inputs_answers = tokenizer(["Hello, my cat is cute", "Hello, my dog is cute"], return_tensors="pt", padding="max_length", max_length=512)
    outputs = model(inputs_questions, inputs_answers, rank_for_one_query=True)
    assert outputs is not None
    assert outputs.shape == torch.Size([2])
    assert outputs[0] > 0 and outputs[0] < 1
    assert outputs[1] > 0 and outputs[1] < 1
    assert outputs[0] != outputs[1]

def test_attention_weights_logging():
    inputs_questions = tokenizer(["Hello, my dog is cute"], return_tensors="pt", padding="max_length", max_length=512)
    inputs_answers = tokenizer(["Hello, my cat is cute"], return_tensors="pt", padding="max_length", max_length=512)
    def callback(weights):
        assert weights is not None
    model = CrossAttentionDistancePredictor("bert-base-uncased", attention_weights_callback=callback)
    outputs = model(inputs_questions, inputs_answers)
    assert outputs is not None
    assert outputs.shape == torch.Size([1])
    assert outputs[0] > 0 and outputs[0] < 1

def test_attention_weights_matplotlib():
    inputs_questions = tokenizer(["Hello, my dog is cute"], return_tensors="pt", padding="max_length", max_length=512)
    inputs_answers = tokenizer(["Hello, my cat is cute"], return_tensors="pt", padding="max_length", max_length=512)
    def callback(weights):
        plt.imshow(weights[0].detach().numpy())
        plt.savefig("test_attention_weights_matplotlib.png")
        assert weights is not None
        assert exists("test_attention_weights_matplotlib.png")
    model = CrossAttentionDistancePredictor("bert-base-uncased", attention_weights_callback=callback)
    outputs = model(inputs_questions, inputs_answers)
    assert outputs is not None
    assert outputs.shape == torch.Size([1])
    assert outputs[0] > 0 and outputs[0] < 1