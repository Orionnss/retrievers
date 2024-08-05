from ...retriever_transformers.retrievers.CrossAttentionRetriever import CrossAttentionRetriever, CrossAttentionRetrieverTrainingArguments, CrossAttentionRetrieverOutput

retriever = CrossAttentionRetriever("bert-base-uncased")

def test_BertEmbedderRetriever_fit():
    queries = ["Hello, my dog is cute", "Hello, my cat is cute", "Dinosaurs are very old animals", "I like to eat pizza", "I like to eat pasta", "I like to eat sushi", "I like to eat burgers", "I like to eat hot dogs", "I like to eat"]
    documents = ["Dogs are the best animals ", "Cats are usually ferocious and independent", "Dinosaurs are extinct", "Pizza is a very popular dish", "Pasta is a very popular dish", "Sushi is a very popular dish", "Burgers are a very popular dish", "Hot dogs are a very popular dish", "I like to eat"]
    losses = []
    loss_callback = lambda loss: losses.append(loss)
    args = CrossAttentionRetrieverTrainingArguments(batch_size=2, shuffle=False, epochs=1, step_callback=loss_callback)
    retriever.fit(queries, documents, args)
    print(losses)
    assert losses[0] > losses[-1]