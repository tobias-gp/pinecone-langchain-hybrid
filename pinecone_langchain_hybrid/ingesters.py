import os

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Text

from pinecone_langchain_hybrid.settings import Settings

class Ingester: 
    """
    Base class for document ingestion. Subclasses should implement the get_documents method.
    """

    def __init__(self):
        pass

    def get_documents(): 
        """
        Method to be implemented by subclasses to retrieve documents.
        """
        raise NotImplementedError("get_documents() must be implemented by subclass")

class PdfIngester(Ingester):
    """
    Class for ingesting PDF documents from a specified directory.
    """

    def __init__(self, path: str, namespace: str) -> None:
        """
        Initialize the PdfIngester with the directory path and namespace.

        Parameters:
        - path (str): The directory path containing PDF files.
        - namespace (str): The namespace for the documents.
        """
        super().__init__()
        self.namespace = namespace
        self.path = path

    def get_documents(self):
        """
        Retrieve and process PDF documents from the specified directory.

        Returns:
        - list: List of processed document chunks with metadata.
        """
        files = [f for f in os.listdir(self.path) if f.endswith('.pdf')]
        all_chunks = []
        chunk_id = 0

        for file in files:
            elements = partition_pdf(os.path.join(self.path, file), strategy="fast")
            chunks = chunk_elements(elements, max_characters=Settings.CHUNKER_PDF_MAX_CHARACTERS, overlap=Settings.CHUNKER_PDF_OVERLAP)

            for chunk in chunks:
                if not isinstance(chunk, Text):
                    continue

                text = str(chunk.text)

                metadata = {
                    "page_number": chunk.metadata.page_number
                }
                
                all_chunks.append({
                    "id":  "document-%s" % chunk_id,
                    "text": text,
                    "metadata": metadata,
                    "namespace": self.namespace
                })

                chunk_id += 1

        return all_chunks