# Callbackable Sentence Transformers

A simple class to replace the `SentenceTransformer` class from [sentence_transformers](https://sbert.net/) package.

It adds a simple parameter `callback_during_training` to the `SentenceTransformer.fit()` function be able to get the loss value of the SentenceTransformer during training.

```python
from callbackable_sentence_transformers import CallbackableSentenceTransformer

model = CallbackableSentenceTransformer(model_checkpoint)

def callback(loss, epoch, step):
    print(f"Loss at epoch {epoch}, {step}: {loss}")

model.fit(..., callback_during_training=callback)
```