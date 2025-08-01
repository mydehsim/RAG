"""
Professional RAG Pipeline - Query Engine Module
===============================================

A comprehensive query engine for Retrieval-Augmented Generation (RAG) applications
with advanced filtering, search optimization, and response generation.

Author: Mustafa Said Oğuztürk
Date: 2025-07-30
Version: 2.0.0
"""

import re
from typing import Dict, List, Any, Tuple, Optional, Union

# LangChain imports
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class QueryEngineConfig:
    """Configuration class for query engine parameters."""
    
    # Database configuration
    CHROMA_PATH = "chroma"
    EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
    EMBEDDING_DEVICE = 'cpu'
    
    # LLM configuration
    LLM_MODEL = "llama3.2"
    LLM_TEMPERATURE = 0.1
    
    # Search configuration
    DEFAULT_K = 5
    MAX_TOC_RESULTS = 1
    
    # Prompt template
    PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
- -
Answer the question based on the above context: {question}
"""


class TableOfContentsFilter:
    """Fast Table of Contents filtering for query results."""
    
    @staticmethod
    def is_toc_quick(content: str) -> bool:
        """
        Ultra-fast ToC detection with minimal processing.
        Only catches the most obvious ToC patterns.
        
        Args:
            content: Text content to analyze
            
        Returns:
            bool: True if content appears to be Table of Contents
        """
        content_lower = content.lower()
        
        # Quick check for explicit ToC headers (first 50 chars only)
        if 'table of contents' in content_lower[:50]:
            return True
        
        # Check for heavy dot patterns (most reliable ToC indicator)
        if '......' in content and len(content) < 200:
            return True
        
        return False
    
    @staticmethod
    def filter_toc_fast(results: List[Tuple], max_toc: int = 1) -> List[Tuple]:
        """
        Fast ToC filtering with minimal processing.
        
        Args:
            results: List of (document, score) tuples
            max_toc: Maximum number of ToC results to keep
            
        Returns:
            List[Tuple]: Filtered results
        """
        non_toc = []
        toc = []
        
        for doc, score in results:
            if TableOfContentsFilter.is_toc_quick(doc.page_content):
                toc.append((doc, score))
            else:
                non_toc.append((doc, score))
        
        # Prefer non-ToC results, but keep some ToC if needed
        if len(non_toc) >= 3:
            return non_toc[:5]  # Top 5 non-ToC
        elif len(non_toc) >= 1:
            return non_toc + toc[:max_toc]  # Add 1 ToC if needed
        else:
            return results[:5]  # Fallback to original if all ToC


class DatabaseAnalyzer:
    """Analyzes database contents and provides file/directory information."""
    
    def __init__(self, chroma_path: str, embeddings):
        """Initialize database analyzer."""
        self.chroma_path = chroma_path
        self.embeddings = embeddings
        self._db = None
    
    @property
    def db(self):
        """Lazy database connection."""
        if self._db is None:
            self._db = Chroma(
                persist_directory=self.chroma_path, 
                embedding_function=self.embeddings
            )
        return self._db
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get comprehensive database information."""
        try:
            all_items = self.db.get(include=["metadatas"])
            
            sources = set()
            directories = set()
            
            for metadata in all_items["metadatas"]:
                if metadata and "source" in metadata:
                    source = metadata["source"]
                    sources.add(source)
                    directory = "/".join(source.split("/")[:-1])
                    if directory:
                        directories.add(directory)
            
            return {
                'files': sorted(list(sources)),
                'directories': sorted(list(directories)),
                'total_chunks': len(all_items["metadatas"])
            }
            
        except Exception as e:
            print(f"Error getting database info: {e}")
            return {'files': [], 'directories': [], 'total_chunks': 0}
    
    def show_database_overview(self) -> None:
        """Display comprehensive database overview."""
        db_info = self.get_database_info()
        
        print("🔍 Database Overview:")
        print(f"  Total chunks: {db_info['total_chunks']}")
        print(f"  Total files: {len(db_info['files'])}")
        print(f"  Total directories: {len(db_info['directories'])}")
        print()
        
        print("📂 Available Directories:")
        for i, directory in enumerate(db_info['directories'], 1):
            files_in_dir = [f for f in db_info['files'] if f.startswith(directory)]
            print(f"  {i:2d}. {directory} ({len(files_in_dir)} files)")
        print()
        
        documents = [f.split('/')[-1] for f in db_info['files']]
        unique_documents = sorted(list(set(documents)))
        
        print("📚 Available Documents:")
        for i, doc in enumerate(unique_documents, 1):
            print(f"  {i:2d}. {doc}")


class ChromaDBFilterBuilder:
    """Builds ChromaDB native filters for document and directory filtering."""
    
    def __init__(self, database_analyzer: DatabaseAnalyzer):
        """Initialize filter builder with database analyzer."""
        self.database_analyzer = database_analyzer
    
    def build_filter(self, document_names: Optional[List[str]] = None, 
                    directory_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Build ChromaDB native filter using supported operators.
        
        Args:
            document_names: List of document names to filter by
            directory_path: Directory path to filter by
            
        Returns:
            Optional[Dict]: ChromaDB filter or None if no filtering needed
        """
        db_info = self.database_analyzer.get_database_info()
        available_files = db_info['files']
        
        # Handle intersection of document names and directory path
        if document_names and directory_path:
            matching_paths = self._find_matching_paths_intersection(
                document_names, directory_path, available_files
            )
        # Handle document names only
        elif document_names:
            matching_paths = self._find_matching_paths_documents(
                document_names, available_files
            )
        # Handle directory path only
        elif directory_path:
            matching_paths = self._find_matching_paths_directory(
                directory_path, available_files
            )
        else:
            return None
        
        return self._create_chroma_filter(matching_paths)
    
    def _find_matching_paths_intersection(self, document_names: List[str], 
                                         directory_path: str, 
                                         available_files: List[str]) -> List[str]:
        """Find files that match both document names and directory path."""
        matching_paths = []
        for doc_name in document_names:
            for file_path in available_files:
                filename = file_path.split('/')[-1]
                if ((filename == doc_name or doc_name in filename) and 
                    file_path.startswith(directory_path)):
                    matching_paths.append(file_path)
        return matching_paths
    
    def _find_matching_paths_documents(self, document_names: List[str], 
                                      available_files: List[str]) -> List[str]:
        """Find files that match document names."""
        matching_paths = []
        for doc_name in document_names:
            for file_path in available_files:
                filename = file_path.split('/')[-1]
                if filename == doc_name or doc_name in filename:
                    matching_paths.append(file_path)
        return matching_paths
    
    def _find_matching_paths_directory(self, directory_path: str, 
                                      available_files: List[str]) -> List[str]:
        """Find files that match directory path."""
        return [f for f in available_files if f.startswith(directory_path)]
    
    def _create_chroma_filter(self, matching_paths: List[str]) -> Optional[Dict[str, Any]]:
        """Create ChromaDB filter from matching paths."""
        if not matching_paths:
            return {"source": {"$eq": "NO_MATCHING_FILES_FOUND"}}
        elif len(matching_paths) == 1:
            return {"source": {"$eq": matching_paths[0]}}
        else:
            return {"source": {"$in": matching_paths}}
    
    def test_filter_building(self, document: Optional[str] = None, 
                           directory: Optional[str] = None) -> Dict[str, Any]:
        """
        Test filter building for debugging purposes.
        
        Args:
            document: Document name to test
            directory: Directory path to test
            
        Returns:
            Dict: Filter building results and debug information
        """
        print(f"🧪 Testing filter building:")
        print(f"   Document: {document}")
        print(f"   Directory: {directory}")
        
        document_names = [document] if document else None
        filter_result = self.build_filter(document_names, directory)
        
        print(f"   Generated filter: {filter_result}")
        
        # Show available files for reference
        db_info = self.database_analyzer.get_database_info()
        print(f"\n📁 Available files ({len(db_info['files'])}):")
        for file in db_info['files'][:10]:  # Show first 10 files
            print(f"   {file}")
        if len(db_info['files']) > 10:
            print(f"   ... and {len(db_info['files']) - 10} more files")
        
        return {
            "filter": filter_result,
            "total_files": len(db_info['files']),
            "directories": db_info['directories']
        }


class ResponseGenerator:
    """Handles LLM initialization and response generation."""
    
    def __init__(self, model_name: str = "llama3.2", temperature: float = 0.1):
        """Initialize response generator with LLM configuration."""
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.prompt_template = ChatPromptTemplate.from_template(
            QueryEngineConfig.PROMPT_TEMPLATE
        )
    
    def generate_response(self, context: str, question: str) -> Any:
        """
        Generate response using LLM.
        
        Args:
            context: Retrieved context from documents
            question: User question
            
        Returns:
            LLM response
        """
        prompt = self.prompt_template.format(context=context, question=question)
        return self.llm.invoke(prompt)
    
    def format_response_output(self, response: Any, sources: List[str]) -> None:
        """
        Format and display response with sources.
        
        Args:
            response: LLM response
            sources: List of source chunk IDs
        """
        print("🧠 Response:")
        print("=" * 50)
        content = response.content if hasattr(response, 'content') else str(response)
        print(content)
        print()
        print("📚 Sources:")
        for i, source in enumerate(sources, 1):
            print(f"  {i}. {source}")


class RAGQueryEngine:
    """Main query engine that orchestrates the search and response generation."""
    
    def __init__(self, chroma_path: str = None, embedding_model: str = None, 
                 llm_model: str = None, llm_temperature: float = None):
        """Initialize RAG query engine."""
        self.config = QueryEngineConfig()
        self.chroma_path = chroma_path or self.config.CHROMA_PATH
        
        # Initialize components
        self.embeddings = self._initialize_embeddings(embedding_model)
        self.database_analyzer = DatabaseAnalyzer(self.chroma_path, self.embeddings)
        self.filter_builder = ChromaDBFilterBuilder(self.database_analyzer)
        self.response_generator = ResponseGenerator(
            model_name=llm_model or self.config.LLM_MODEL,
            temperature=llm_temperature or self.config.LLM_TEMPERATURE
        )
        self.toc_filter = TableOfContentsFilter()
    
    def _initialize_embeddings(self, embedding_model: Optional[str] = None):
        """Initialize embedding function."""
        model_name = embedding_model or self.config.EMBEDDING_MODEL
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.config.EMBEDDING_DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def search_documents(self, query: str, k: int = None,
                        document_names: Optional[List[str]] = None,
                        directory_path: Optional[str] = None,
                        enable_toc_filter: bool = True) -> Tuple[str, Any]:
        """
        Core search function with filtering and response generation.
        
        Args:
            query: Search query
            k: Number of results to retrieve
            document_names: List of document names to filter by
            directory_path: Directory path to filter by
            enable_toc_filter: Whether to filter Table of Contents
            
        Returns:
            Tuple[str, Any]: Formatted response and raw LLM response
        """
        k = k or self.config.DEFAULT_K
        
        # Build ChromaDB filter
        where_filter = self.filter_builder.build_filter(document_names, directory_path)
        
        # Perform search
        db = self.database_analyzer.db
        if where_filter:
            results = db.similarity_search_with_relevance_scores(query, k=k, filter=where_filter)
        else:
            results = db.similarity_search_with_relevance_scores(query, k=k)
        
        if not results:
            no_results_msg = "No matching documents found."
            return no_results_msg, no_results_msg
        
        # Apply ToC filtering if enabled
        if enable_toc_filter:
            results = self.toc_filter.filter_toc_fast(results, max_toc=self.config.MAX_TOC_RESULTS)
        
        if not results:
            filtered_msg = "No documents match after filtering."
            return filtered_msg, filtered_msg
        
        # Generate response
        context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
        response = self.response_generator.generate_response(context_text, query)
        
        # Format sources
        sources = [doc.metadata.get("id", "Unknown") for doc, _score in results]
        
        return sources, response
    
    def quick_search(self, query: str, document: Optional[str] = None, 
                    directory: Optional[str] = None, k: int = None) -> Any:
        """
        🎯 MAIN SEARCH FUNCTION - Handles all filtering scenarios.
        
        Args:
            query: Search query
            document: Single document name to search in
            directory: Directory path to search in
            k: Number of results to retrieve
            
        Returns:
            LLM response
        """
        document_names = [document] if document else None
        sources, response = self.search_documents(
            query=query,
            document_names=document_names,
            directory_path=directory,
            k=k
        )
        
        # Display results
        if isinstance(sources, str):  # Error message
            print(f"❌ {sources}")
            return response
        
        self.response_generator.format_response_output(response, sources)
        return response
    
    def search_multiple_documents(self, query: str, *document_names: str, k: int = None) -> Any:
        """
        🎯 SEARCH IN MULTIPLE DOCUMENTS.
        
        Args:
            query: Search query
            *document_names: Variable number of document names
            k: Number of results to retrieve
            
        Returns:
            LLM response
        """
        if not document_names:
            return self.quick_search(query, k=k)
        
        sources, response = self.search_documents(
            query=query,
            document_names=list(document_names),
            k=k
        )
        
        # Display results
        if isinstance(sources, str):  # Error message
            print(f"❌ {sources}")
            return response
        
        self.response_generator.format_response_output(response, sources)
        return response
    
    def show_database_info(self) -> None:
        """Show database overview."""
        self.database_analyzer.show_database_overview()
    
    def test_filter(self, document: Optional[str] = None, 
                   directory: Optional[str] = None) -> Dict[str, Any]:
        """Test filter building for debugging."""
        return self.filter_builder.test_filter_building(document, directory)


# Convenience functions for easy usage
def create_query_engine(chroma_path: str = None, embedding_model: str = None,
                       llm_model: str = None, llm_temperature: float = None) -> RAGQueryEngine:
    """Create and return a configured query engine."""
    return RAGQueryEngine(
        chroma_path=chroma_path,
        embedding_model=embedding_model,
        llm_model=llm_model,
        llm_temperature=llm_temperature
    )


def quick_search(query: str, document: Optional[str] = None, 
                directory: Optional[str] = None, k: int = 5) -> Any:
    """
    Quick search function with default configuration.
    
    Examples:
        quick_search("radar test")                                    # No filter
        quick_search("radar test", document="document.pdf")           # One document
        quick_search("radar test", directory="/kaggle/input/srs")     # One directory
        quick_search("radar test", document="doc.pdf", directory="/kaggle/input")  # Both
    """
    engine = create_query_engine()
    return engine.quick_search(query, document, directory, k)


def search_multiple_docs(query: str, *document_names: str, k: int = 5) -> Any:
    """
    Search in multiple documents.
    
    Example:
        search_multiple_docs("radar test", "doc1.pdf", "doc2.docx", "doc3.pdf")
    """
    engine = create_query_engine()
    return engine.search_multiple_documents(query, *document_names, k=k)


def show_database_info() -> None:
    """Show database overview."""
    engine = create_query_engine()
    engine.show_database_info()


def test_filter(document: Optional[str] = None, directory: Optional[str] = None) -> Dict[str, Any]:
    """Test filter building for debugging."""
    engine = create_query_engine()
    return engine.test_filter(document, directory)


if __name__ == "__main__":
    # Example usage
    print("🚀 RAG Query Engine Ready!")
    print("\n📖 Usage Examples:")
    print("1. quick_search('What is radar testing?')")
    print("2. quick_search('system requirements', document='SRS.pdf')")
    print("3. quick_search('test procedures', directory='/kaggle/input/tests')")
    print("4. search_multiple_docs('performance metrics', 'doc1.pdf', 'doc2.docx')")
    print("5. show_database_info()")
