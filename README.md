# pinecone-langchain-hybrid
LangChain Retrieval Tool for Pinecone Hybrid Search with Ingest

## Prerequisites

A Pinecone (https://pinecone.io) index with dimensions 3072 and dotproduct for similarity. You can use the free starter subscription. 

An OpenAI API key with access to GPT-4o. If you are using Azure, you will have to modify the instantiation of OpenAIEmbeddings.

## Preparation

Create a new virtual env or conda env and install dependencies 

pip install -r requirements.txt

Set environment variables: 
PINECONE_INDEX = "index_name"
PINECONE_API_KEY = "..."
OPENAI_API_KEY = "..."

## Ingester 

Upload chunks 

python -m pinecone_langchain_hybrid.uploader

Modify the ingester to add additional parsers or folders. Currently, there's only one default parser given as an example. 

## Retriever

Import the retriever in your own project or run

simply modify the main method and call: 

python -m pinecone_langchain_hybrid.retrievers

you can also use it in Python by writing: 

tool = DocumentsPineconeRetrieverTool()

input = {
    "query": "What is AI?"
}

documents_prompt = tool.invoke(input=input)

logging.info(documents_prompt)