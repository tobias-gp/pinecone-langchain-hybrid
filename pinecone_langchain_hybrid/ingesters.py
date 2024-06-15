import os

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.basic import chunk_elements
from unstructured.documents.elements import Text

from pinecone_langchain_hybrid.settings import Settings

class Ingester: 

    def __init__(self):
        pass

    def get_documents(): 
        raise NotImplementedError("get_documents() must be implemented by subclass")

class PdfIngester(Ingester):

    def __init__(self, path: str, namespace: str) -> None:
        super().__init__()
        self.namespace = namespace
        self.path = path

    def get_documents(self):
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
