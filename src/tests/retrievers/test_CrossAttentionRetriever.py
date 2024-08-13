from ...retriever_transformers.retrievers.CrossAttentionRetriever import CrossAttentionRetriever, CrossAttentionRetrieverTrainingArguments, CrossAttentionRetrieverOutput

retriever = CrossAttentionRetriever("bert-base-uncased", seed=43)

def test_CrossAttentionRetriever_fit():
    queries = ["Words to trigger damages", "The sea is blue"]
    documents = ["Words to trigger damages ", "The sea is blue"]
    losses = []
    loss_callback = lambda loss: losses.append(loss)
    num_epochs = 50
    batch_size = 2
    args = CrossAttentionRetrieverTrainingArguments(batch_size=batch_size, shuffle=False, epochs=num_epochs, step_callback=loss_callback, learning_rate=1e-5)
    retriever.fit(queries, documents, args)
    assert losses[0] > losses[-1]

def test_CrossAttentionRetriever_fit_tqdm():
    queries = ["Words to trigger damages", "The sea is blue"]
    documents = ["Words to trigger damages ", "The sea is blue"]
    losses = []
    loss_callback = lambda loss: losses.append(loss)
    num_epochs = 2
    batch_size = 2
    args = CrossAttentionRetrieverTrainingArguments(batch_size=batch_size, shuffle=False, epochs=num_epochs, step_callback=loss_callback, learning_rate=1e-5)
    retriever.fit(queries, documents, args, progress_bar=True)

def test_CrossAttentionRetriever_rank():
    queries = ["Words to trigger damages", "The sea is blue"]
    documents = ["Words to trigger damages ", "The sea is blue"]
    ranks = retriever.rank(queries, documents)
    assert ranks is not None
    assert len(ranks) == 2
    assert len(ranks[0]) == 2

def test_CrossAttentionRetriever_compute_mrr_and_accuracy():
    queries = ["Words to trigger damages", "The sea is blue"]
    documents = ["Words to trigger damages ", "The sea is blue"]
    ranks = [[0.8, 0.2], [0.4, 0.6]]
    output = retriever.compute_mrr_and_accuracy(ranks)
    assert output.mrr is not None
    assert output.accuracy is not None
    assert output.mrr == 1.0
    assert output.accuracy == 1.0

