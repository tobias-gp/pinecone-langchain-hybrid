from typing import Optional, Type, List
import json
import os
import logging

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import numpy as np

from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers.pinecone_hybrid_search import PineconeHybridSearchRetriever as PineconeHybridSearchRetrieverOriginal
from langchain_core.prompts import format_document, PromptTemplate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

from pinecone_text.sparse import BM25Encoder
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone import Pinecone

import nltk

from pinecone_langchain_hybrid.settings import Settings

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models")
BM25_PATH = os.path.join(MODEL_PATH, "bm25_values.json")

pinecone = Pinecone(api_key=PINECONE_API_KEY)

# specify the path if working with a non-writable file system and download NLTK deps beforehand
# nltk.data.path = [MODEL_PATH]

class PineconeHybridSearchRetriever(PineconeHybridSearchRetrieverOriginal):

    def _get_relevant_documents(
            self, 
            query: str, 
            *, 
            run_manager: CallbackManagerForRetrieverRun,
            filter: dict = {}, 
            top_k: int = None) -> List[Document]:
        """
        Overriding this from parent class to support fast non-embedding search with alpha = 0 
        We also add the ability to set filters for metadata
        """
        import time

        sparse_vec = self.sparse_encoder.encode_queries(query)

        if top_k is None: 
            top_k = self.top_k

        if self.alpha == 0:
            dense_vec = np.zeros(Settings.EMBEDDING_DIMENSION).tolist()
        else: 
            dense_vec = self.embeddings.embed_query(query)

        dense_vec, sparse_vec = hybrid_convex_scale(dense_vec, sparse_vec, self.alpha)
        sparse_vec["values"] = np.array(sparse_vec["values"], dtype=float).tolist()

        start = time.time()

        result = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=top_k,
            include_metadata=True,
            include_values=True,
            namespace=self.namespace,
            filter=filter
        )

        logging.info("Pinecone query time: %s" % (time.time() - start))

        final_result = []
        for res in result["matches"]:
            if res["score"] < Settings.SCORE_RELEVANCE_THRESHOLD:
                continue

            context = res["metadata"].pop("context")
            metadata = res["metadata"]
            metadata["score"] = res["score"]

            final_result.append(Document(page_content=context, metadata=metadata))

        return final_result

class PineconeRetrieverTool(BaseTool):
    namespace: str
    alpha: float = 0.5
    top_k: int = 10

    @property
    def retriever(self):
        index = pinecone.Index(PINECONE_INDEX)

        bm25_encoder = BM25Encoder(language=Settings.NLTK_LANGUAGE).load(BM25_PATH)
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=Settings.OPENAI_EMBEDDING_MODEL)
        retriever = PineconeHybridSearchRetriever(
            alpha=self.alpha, 
            embeddings=embeddings, 
            sparse_encoder=bm25_encoder, 
            index=index, 
            namespace=self.namespace,
            top_k=self.top_k,
        )

        return retriever
        
    def retrieve_documents(self, query: str, **kwargs) -> list:
        docs = self.retriever.invoke(query)

        return docs

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None, **kwargs) -> str:
        document_separator = "\n\n"
        document_prompt = PromptTemplate.from_template('<meta page_number="{page_number:.0f}" />\n\n{page_content}')

        docs = self.retrieve_documents(query, **kwargs)

        if len(docs) == 0:
            return {"result": "No results found." }

        return document_separator.join(
            format_document(doc, document_prompt) for doc in docs
        )
        
    async def _arun(self, **kwargs) -> str:
        return self._run(**kwargs)

class DocumentsPineconeRetrieverToolArgsSchema(BaseModel):
    query: str = Field(default=None, description="The query to search for in the knowledge base. The query can contain one or multiple search words or a question.")
    
class DocumentsPineconeRetrieverTool(PineconeRetrieverTool):
    name: str = "documents_pinecone_retriever_tool"
    description: str = (
        "Call this tool to search for information in the knowledge base"
    )
    args_schema: Type[DocumentsPineconeRetrieverToolArgsSchema] = DocumentsPineconeRetrieverToolArgsSchema
    namespace: str = "default"

if __name__ == "__main__":
    tool = DocumentsPineconeRetrieverTool(top_k=5, alpha=0.5)

    input = {
        "query": "Which features can I use for emotion detection?"
    }

    import time
    start_time = time.time()

    documents_prompt = tool.invoke(input=input)

    print("Time taken: %s" % (time.time() - start_time))

    #print(documents_prompt)