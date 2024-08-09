from ...retriever_transformers.retrievers.CrossAttentionRetriever import CrossAttentionRetriever, CrossAttentionRetrieverTrainingArguments, CrossAttentionRetrieverOutput

retriever = CrossAttentionRetriever("bert-base-uncased", seed=43)

def test_CrossAttentionRetriever_fit():
    queries = ["Words to trigger damages", "The sea is blue"]
    documents = ["Words to trigger damages ", "The sea is blue"]
    losses = []
    loss_callback = lambda loss: losses.append(loss)
    num_epochs = 10
    batch_size = 2
    args = CrossAttentionRetrieverTrainingArguments(batch_size=batch_size, shuffle=False, epochs=num_epochs, step_callback=loss_callback, learning_rate=1e-5)
    retriever.fit(queries, documents, args)
    assert losses[0] > losses[-1]

def test_CrossAttentionRetriever_fit_tqdm():
    queries = ["Words to trigger damages", "The sea is blue"]
    documents = ["Words to trigger damages ", "The sea is blue"]
    losses = []
    loss_callback = lambda loss: losses.append(loss)
    num_epochs = 10
    batch_size = 2
    args = CrossAttentionRetrieverTrainingArguments(batch_size=batch_size, shuffle=False, epochs=num_epochs, step_callback=loss_callback, learning_rate=1e-5)
    retriever.fit(queries, documents, args, progress_bar=True)
    assert losses[0] > losses[-1]
