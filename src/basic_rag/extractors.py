from abc import ABC
from abc import abstractmethod
from io import BytesIO

from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter


class Extractor(ABC):
    """Base class for extracting text from documents."""
    @abstractmethod
    def extract(self, document: str) -> str:
        """Extract text from a document."""
        pass


class DoclingExtractor(Extractor):
    """Extractor that uses Docling to extract text from documents."""

    def __init__(self) -> None:
        self.converter = DocumentConverter()

    def extract(self, filename, document: bytes) -> str:
        """Extract text from a document using Docling."""
        buf = BytesIO(document)
        source = DocumentStream(name=filename, stream=buf)
        result = self.converter.convert(source)
        return result.document.export_to_markdown()


docling_extractor = DoclingExtractor()
