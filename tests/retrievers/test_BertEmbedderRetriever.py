from ...src.retriever_transformers.retrievers.BertEmbedderRetriever import BertEmbedderRetriever, EmbedderRetrieverTrainingArguments, EmbedderRetrieverOutput

retriever = BertEmbedderRetriever("bert-base-uncased")

def test_BertEmbedderRetriever():
    query = "Hello, my dog is cute"
    document = "Hello, my cat is cute"
    embedded_query = retriever._encode(query)
    embedded_document = retriever._encode(document)
    assert embedded_query is not None
    assert embedded_document is not None
    assert embedded_query.shape == (1, 768)
    assert embedded_document.shape == (1, 768)

def test_BertEmbedderRetriever_fit():
    queries = ["Hello, my dog is cute", "Hello, my cat is cute", "Dinosaurs are very old animals", "I like to eat pizza", "I like to eat pasta", "I like to eat sushi", "I like to eat burgers", "I like to eat hot dogs", "I like to eat"]
    documents = ["Dogs are the best animals ", "Cats are usually ferocious and independent", "Dinosaurs are extinct", "Pizza is a very popular dish", "Pasta is a very popular dish", "Sushi is a very popular dish", "Burgers are a very popular dish", "Hot dogs are a very popular dish", "I like to eat"]
    losses = []
    loss_callback = lambda loss: losses.append(loss)
    args = EmbedderRetrieverTrainingArguments(batch_size=2, shuffle=False, epochs=1, step_callback=loss_callback)
    retriever.fit(queries, documents, args, margin=0.01)
    assert losses[0] > losses[1]

def test_BertEmbedderRetriever_evaluate():
    queries = ["Hello, my dog is cute", "Hello, my cat is cute", "Dinosaurs are very old animals", "I like to eat pizza", "I like to eat pasta", "I like to eat sushi", "I like to eat burgers", "I like to eat hot dogs", "I like to eat"]
    documents = ["Dogs are the best animals ", "Cats are usually ferocious and independent", "Dinosaurs are extinct", "Pizza is a very popular dish", "Pasta is a very popular dish", "Sushi is a very popular dish", "Burgers are a very popular dish", "Hot dogs are a very popular dish", "I like to eat"]
    metrics = retriever.evaluate(queries, documents)
    assert metrics is not None
    assert type(metrics) == EmbedderRetrieverOutput
    assert metrics.mrr >= 0 and metrics.mrr <= 1
    assert metrics.accuracy >= 0 and metrics.accuracy <= 1