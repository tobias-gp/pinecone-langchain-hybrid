import os
import logging
import sys
from typing import List
import json
import hashlib

from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
import nltk
from nltk.tokenize import word_tokenize
import html2text

from pinecone_langchain_hybrid.settings import Settings
from pinecone_langchain_hybrid.ingesters import PdfIngester

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX = os.environ.get("PINECONE_INDEX")

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
PINECONE_BATCH_SIZE = 32
MIN_TEXT_LENGTH = 1000

MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../models")
BM25_PATH = os.path.join(MODEL_PATH, "bm25_values.json")

# specify the path if working with a non-writable file system and download NLTK deps beforehand
# nltk.data.path = [MODEL_PATH]

# Modify this if using Azure OpenAI 
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=Settings.OPENAI_EMBEDDING_MODEL)
pinecone = Pinecone(api_namespace=PINECONE_API_KEY)

class Uploader:
    """
    Base class for uploading documents. Subclasses should implement specific upload methods.
    """

    def __init__(self) -> None:
        pass

class PineconeUploader(Uploader):
    """
    Class for uploading documents to Pinecone.
    """

    def __init__(self, ingesters) -> None:
        """
        Initialize the PineconeUploader with a list of ingesters.

        Parameters:
        - ingesters (List[Ingester]): List of document ingesters.
        """
        self.documents = []
        self.ingesters = ingesters

    def run(self):
        """
        Main method to run the uploader process: prepare the index, get documents, update the corpus, and upload documents.
        """
        self.prepare_index()
        self.get_documents()
        self.update_corpus()
        self.upload()

    def get_documents(self):
        """
        Retrieve documents from all ingesters and store them in the self.documents list.
        """
        for ingester in self.ingesters:
            documents = ingester.get_documents()

            logging.info("Got %i documents from ingester %s" % (len(documents), ingester.__class__.__name__))
            self.documents.extend(documents)

    def prepare_index(self):
        """
        Prepare the Pinecone index for document insertion.
        """
        self.index = pinecone.Index(PINECONE_INDEX)

    def update_corpus(self): 
        """
        Update the corpus by tokenizing the text and fitting the BM25 encoder. Save the BM25 values to a file.
        """
        corpus = ""

        for document in self.documents:
            corpus += document["text"] + "\n\n"

        corpus = word_tokenize(corpus, language=Settings.NLTK_LANGUAGE)

        # save corpus
        corpus_path = os.path.join(MODEL_PATH, "bm25_corpus.txt")
        with open(corpus_path, 'w') as f:
            f.write('\n'.join(corpus))

        logging.info("Fitting BM25 encoder")
        bm25_encoder = BM25Encoder(language=Settings.NLTK_LANGUAGE)
        bm25_encoder.fit(corpus)

        logging.info("Saving BM25 values to %s" % BM25_PATH)
        bm25_encoder.dump(BM25_PATH)

    def upload(self): 
        """
        Upload all documents to the Pinecone index, organized by namespaces.
        """
        # delete all documents in the target namespaces
        namespaces = [] 
        for document in self.documents:
            if document["namespace"] not in namespaces:
                namespaces.append(document["namespace"])

        for namespace in namespaces:
            logging.info("Uploading to namespace %s" % namespace)
            self.upload_to_namespace(namespace)

    def upload_to_namespace(self, namespace: str):
        """
        Upload documents to a specific namespace in Pinecone.

        Parameters:
        - namespace (str): The target namespace.
        """
        bm25 = BM25Encoder().load(BM25_PATH)
        index = pinecone.Index(PINECONE_INDEX)

        texts_batch = []
        ids_batch = []
        metadatas_batch = []
        i = 0

        if namespace in index.describe_index_stats()['namespaces']:
            index.delete(delete_all=True, namespace=namespace)

        # filter documents for namespaces
        documents = [document for document in self.documents if document["namespace"] == namespace]

        logging.info("Uploading %i documents to namespace %s" % (len(documents), namespace))

        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings, 
            sparse_encoder=bm25, 
            index=index, 
            namespace=namespace
        )

        # For each file in the directory
        for document in documents:
            texts_batch.append(document["text"])
            ids_batch.append(document["id"])

            if "metadata" in document:
                metadatas_batch.append(document["metadata"])
            else: 
                metadatas_batch.append({})

            i += 1

            # for every PINECONE_BATCH_SIZE files, upsert into db
            if len(texts_batch) >= PINECONE_BATCH_SIZE:
                logging.info("Inserting documents from %i to %i" % (i - PINECONE_BATCH_SIZE, i))
                retriever.add_texts(texts=texts_batch, ids=ids_batch, metadatas=metadatas_batch, namespace=namespace)
                texts_batch = []
                ids_batch = []

        retriever.add_texts(texts=texts_batch, ids=ids_batch, metadatas=metadatas_batch, namespace=namespace)
        logging.info("Inserting documents from %i to %i" % (i - PINECONE_BATCH_SIZE, i))

if __name__ == "__main__":
    uploader = PineconeUploader(
        ingesters=[
            PdfIngester(path=os.path.join(SCRIPT_PATH, "../documents"), namespace="default")
        ]
    )
    uploader.run()