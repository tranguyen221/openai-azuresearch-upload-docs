import logging
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from langchain.schema import Document
import PyPDF2

class FormRecognizerLoader:
    """
    Load a FPD document using the Form Recognizer service and return a list of Document object.
    """
    def __init__(self, endpoint, key, bytes_pdf):
        self.endpoint = endpoint
        self.key = key
        self.bytes_pdf = bytes_pdf
        self.fr_analyser = DocumentAnalysisClient(endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key))
        
    def load(self):
        try:
            poller = self.fr_analyser.begin_analyze_document(
                "prebuilt-read", 
                self.bytes_pdf)
        except HttpResponseError as error:
            if error.error is not None:
                if error.error.code == "InvalidImage":
                    logging.error(f"Received an invalid image error: {error.error}")
                elif error.error.code == "InvalidRequest":
                    logging.error(f"Received an invalid request error: {error.error}")
                elif error.error.code == "InvalidContentLength":
                    logging.error(f"Received an invalid request error: {error.error}")
        result = poller.result()

        documents = [
            Document(
                page_content="\n".join([l.content for l in p.lines]),
                metadata={
                    # "source": _self.doc_path,
                    "page": i+1,
                },
            )
            for i, p in enumerate(result.pages)
        ]

        return documents
    
    def validate_page_count(self, result):
        """Validate the page count of the Form Recognizer result against the PyPDF2 page count
        to make sure pages are not rejected due to the use of lower tiers of the Form Recognizer service.
        Log an error if the page count does not match."""
        pypdf_page_count = len(PyPDF2.PdfReader(self.doc_path).pages)
        if pypdf_page_count != len(result.pages):
            logging.error(
                f"Page count mismatch: PyPDF2 found {pypdf_page_count} pages, but Form Recognizer found {len(result.pages)} pages."
            )