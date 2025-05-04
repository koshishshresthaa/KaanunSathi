
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

class DocumentHandler:
    """
    Handles loading and splitting of documents into chunksx.
    """
    def __init__(self, file_path):
        """
        Initialize with file paths and embedding model name.
        """
        self.file_path = file_path
        
    def load_documents(self):
        """
        Load documents from file paths.
        """
        loader = PyPDFLoader(self.file_path)
        docs = loader.load()
        # Concatenate the content of all pages
        page_content = ""
        for doc in docs:
            page_content += doc.page_content + "\n"  

        return page_content
    
    # Step 2: Split Text by Chapters
    def split_text_by_chapters(self):
        """
        Split the text of the pages by its chapters.
        """
        chapter_pattern = r'Chapter\s*-\s*\d+\s*(.*?)(?=Chapter\s*-\s*\d+\s*|\Z)'
        text=self.load_documents()
        chapters = re.findall(chapter_pattern, text, re.DOTALL)
        return chapters
