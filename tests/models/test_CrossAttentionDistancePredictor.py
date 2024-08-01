from ...src.models.CrossAttentionDistancePredictor import CrossAttentionDistancePredictor
from transformers import AutoTokenizer
import torch

model = CrossAttentionDistancePredictor("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def test_SimpleDistancePrediction():
    inputs_questions = tokenizer(["Hello, my dog is cute"], return_tensors="pt", padding="max_length", max_length=512, )
    inputs_answers = tokenizer(["Hello, my cat is cute"], return_tensors="pt", padding="max_length", max_length=512)
    outputs = model(inputs_questions, inputs_answers)
    assert outputs is not None
    assert outputs.shape == torch.Size([1, 1])
    assert outputs[0] > 0 and outputs[0] < 1