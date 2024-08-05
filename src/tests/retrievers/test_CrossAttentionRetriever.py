from ...retriever_transformers.retrievers.CrossAttentionRetriever import CrossAttentionRetriever, CrossAttentionRetrieverTrainingArguments, CrossAttentionRetrieverOutput

retriever = CrossAttentionRetriever("bert-base-uncased", seed=42)

def test_CrossAttentionRetriever_fit():
    queries = ["Hello, my dog is cute", "Hello, my cat is cute"]
    documents = ["Dogs are the best animals ", "Cats are usually ferocious and independent"]
    losses = []
    loss_callback = lambda loss: losses.append(loss)
    num_epochs = 2
    batch_size = 2
    args = CrossAttentionRetrieverTrainingArguments(batch_size=batch_size, shuffle=False, epochs=num_epochs, step_callback=loss_callback, learning_rate=1e-10)
    retriever.fit(queries, documents, args)
    print(losses)
    print(losses[0], losses[-1])
    assert len(losses) == num_epochs * (len(queries) // batch_size)
    assert losses[0] > losses[-1]