# LangChain Retrieval Tool for Pinecone Hybrid Search with Ingest

I couldn't find a lot of supporting information when implementing a hybrid search retriever for the vector database service Pinecone in combination with LangChain. Even if you are not using LangChain, you may use the tool as a standalone version (see below). 

I hope that this example helps others to implement a high-quality semantic search! Get in touch to buy me a coffee ;) 

## Prerequisites

- A Pinecone index with dimensions 3072 and dotproduct for similarity. You can use the free starter subscription.
- An OpenAI API key with access to GPT-4. If you are using Azure, you will have to modify the instantiation of OpenAIEmbeddings.

## Preparation

1. Create a new virtual environment or conda environment and install dependencies:

```sh
pip install -r requirements.txt
```

2. Set environment variables:

```sh
export PINECONE_INDEX="index_name"
export PINECONE_API_KEY="your_pinecone_api_key"
export OPENAI_API_KEY="your_openai_api_key"
```

## Ingester

The folder documents already contains an example document. To upload chunks, run: 

```sh
python -m pinecone_langchain_hybrid.uploader
```

You can modify the ingester to add additional parsers or folders. Currently, there's only one default parser provided as an example.

## Retriever

### Importing the Retriever

You can import the retriever in your own project or run it directly.

### Running the Retriever

This is just for testing, you can modify the main method and call:

```sh
python -m pinecone_langchain_hybrid.retrievers
```

### Using the Retriever in Python

You can use the retriever in Python by importing the corresponding LangChain tool:

```python
from pinecone_langchain_hybrid.retrievers import DocumentsPineconeRetrieverTool

tool = DocumentsPineconeRetrieverTool()
input = {
    "query": "Which features can I use for emotion detection?"
}

documents_prompt = tool.invoke(input=input)
print(documents_prompt)
```

Alternatively, you can use the retriever directly by importing `PineconeRetrieverTool` and calling `retrieve_documents`:

```python
from pinecone_langchain_hybrid.retrievers import DocumentsPineconeRetrieverTool

tool = DocumentsPineconeRetrieverTool(top_k=5, alpha=0.5)
docs = tool.retrieve_documents(query="Which features can I use for emotion detection?")

print(docs)
```

## Contributing

Feel free to submit issues or pull requests if you find any bugs or have feature requests.

## License

This project is licensed under the MIT License. See the LICENSE file for details.