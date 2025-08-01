"""
Professional RAG Pipeline - Main Setup & Configuration
======================================================

Main orchestration module for the RAG pipeline with easy-to-use functions
and comprehensive configuration management.

Author: Mustafa Said Oğuztürk
Date: 2025-07-30
Version: 2.0.0
"""

import os
import subprocess
import warnings
from pathlib import Path
from typing import Optional, Dict, Any

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class RAGPipelineConfig:
    """Central configuration for the entire RAG pipeline."""
    
    # =============================================================================
    # CORE PATHS
    # =============================================================================
    DATA_PATH = r"C:\Users\stajyer1\Desktop\Mustafa_Said_Oguzturk\data"  # Change this to your document directory
    CHROMA_PATH = "chroma"       # Vector database storage path
    
    # =============================================================================
    # MODEL CONFIGURATIONS
    # =============================================================================
    
    # Embedding Model Options (choose one):
    EMBEDDING_MODELS = {
        "multilingual": "intfloat/multilingual-e5-large",          # Best for multilingual (Turkish + English)
        "english": "sentence-transformers/all-MiniLM-L6-v2",       # Faster, English-focused
        "turkish": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",  # Turkish-specific
        "performance": "SMARTICT/multilingual-e5-large-wiki-tr-rag"     # Optimized for Turkish RAG
    }
    CURRENT_EMBEDDING_MODEL = EMBEDDING_MODELS["multilingual"]
    EMBEDDING_DEVICE = 'cpu'  # Use 'cpu' if no GPU available | Use 'cuda' to accelerate
    
    # LLM Model Options:
    LLM_MODELS = {
        "llama3.2": {"model": "llama3.2", "temp": 0.1},                               # Default, good balance
        "qwen3": {"model": "qwen3", "temp": 0.1},                                     # Alternative
        "nomic-embed-text": {"model": "nomic-embed-text", "temp": 0.1},               # Fallback option
    }
    CURRENT_LLM = LLM_MODELS["llama3.2"]
    
    # =============================================================================
    # PROCESSING PARAMETERS
    # =============================================================================
    
    # Text Chunking
    CHUNK_SIZE = 800        # Characters per chunk      |    alternative 1000
    CHUNK_OVERLAP = 400     # Overlap between chunks    |    alternative 200
    
    # Search Parameters
    DEFAULT_K = 8           # Number of results to retrieve     |    alternative 5
    MAX_TOC_RESULTS = 1     # Maximum Table of Contents results to keep
    
    # File Processing
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.doc', '.md', '.html', '.htm', '.rtf', '.odt', '.xlsx', '.xls'}
    
    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================
    ENABLE_TOC_FILTERING = False #True      # Filter Table of Contents automatically
    ENABLE_OCR = False               # OCR for scanned documents (slower)
    ENABLE_TABLE_EXTRACTION = True   # Extract tables from documents
    ENABLE_COMPREHENSIVE_LOGGING = False  # Detailed processing logs


class SystemSetup:
    """Handles system setup and Ollama installation/configuration."""
    
    @staticmethod
    def install_ollama() -> bool:
        """Install Ollama if not already installed."""
        try:
            # Check if already installed
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Ollama is already installed")
                return True
        except FileNotFoundError:
            pass
        
        print("📦 Installing Ollama...")
        try:
            # Install Ollama
            install_result = subprocess.run(
                ["curl", "-fsSL", "https://ollama.com/install.sh"], 
                capture_output=True, text=True
            )
            if install_result.returncode == 0:
                subprocess.run(["sh"], input=install_result.stdout, text=True)
                print("✅ Ollama installed successfully")
                return True
            else:
                print("❌ Failed to install Ollama")
                return False
        except Exception as e:
            print(f"❌ Error installing Ollama: {e}")
            return False
    
    @staticmethod
    def start_ollama_server() -> subprocess.Popen:
        """Start Ollama server in background."""
        print("🚀 Starting Ollama server...")
        try:
            process = subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            print("✅ Ollama server started")
            return process
        except Exception as e:
            print(f"❌ Error starting Ollama server: {e}")
            return None
    
    @staticmethod
    def pull_llm_model(model_name: str = "llama3.2") -> bool:
        """Pull LLM model if not already available."""
        print(f"📥 Pulling model: {model_name}")
        try:
            result = subprocess.run(
                ["ollama", "pull", model_name], 
                capture_output=True, 
                text=True
            )
            if result.returncode == 0:
                print(f"✅ Model {model_name} ready")
                return True
            else:
                print(f"❌ Failed to pull model {model_name}")
                print(f"Error: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error pulling model: {e}")
            return False
    
    @staticmethod
    def setup_complete_system() -> bool:
        """Complete system setup including Ollama and model."""
        print("🔧 Setting up RAG Pipeline System...")
        print("=" * 50)
        
        # Install Ollama
        if not SystemSetup.install_ollama():
            return False
        
        # Start server
        server_process = SystemSetup.start_ollama_server()
        if server_process is None:
            return False
        
        # Pull model
        model_name = RAGPipelineConfig.CURRENT_LLM["model"]
        if not SystemSetup.pull_llm_model(model_name):
            return False
        
        print("=" * 50)
        print("✅ System setup complete!")
        return True


class RAGPipeline:
    """Main RAG Pipeline orchestrator that combines all components."""
    
    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        """Initialize RAG Pipeline with configuration."""
        self.config = config or RAGPipelineConfig()
        self.document_processor = None
        self.query_engine = None
        self._ollama_process = None
    
    def setup_system(self) -> bool:
        """Setup the complete system."""
        return SystemSetup.setup_complete_system()
    
    def initialize_document_processor(self):
        """Initialize document processor lazily."""
        if self.document_processor is None:
            from document_processing import RAGDocumentProcessor
            self.document_processor = RAGDocumentProcessor(
                data_path=self.config.DATA_PATH,
                chroma_path=self.config.CHROMA_PATH
            )
        return self.document_processor
    
    def initialize_query_engine(self):
        """Initialize query engine lazily."""
        if self.query_engine is None:
            from query_engine import RAGQueryEngine
            self.query_engine = RAGQueryEngine(
                chroma_path=self.config.CHROMA_PATH,
                embedding_model=self.config.CURRENT_EMBEDDING_MODEL,
                llm_model=self.config.CURRENT_LLM["model"],
                llm_temperature=self.config.CURRENT_LLM["temp"]
            )
        return self.query_engine
    
    # =========================================================================
    # DOCUMENT PROCESSING METHODS
    # =========================================================================
    
    def build_database(self, force_rebuild: bool = False) -> None:
        """
        Build or rebuild the document database.
        
        Args:
            force_rebuild: If True, completely rebuild database from scratch
        """
        processor = self.initialize_document_processor()
        
        if force_rebuild:
            print("⚠️ REBUILDING DATABASE FROM SCRATCH")
            # Here you would add database wiping logic
            processor.build_initial_database(comprehensive_toc_filter=True)
        else:
            print("🔄 UPDATING DATABASE")
            processor.update_database(comprehensive_toc_filter=True)
    
    def update_database(self) -> Dict[str, int]:
        """Update database with new/modified files."""
        processor = self.initialize_document_processor()
        return processor.update_database(comprehensive_toc_filter=True)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        engine = self.initialize_query_engine()
        return engine.database_analyzer.get_database_info()
    
    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    
    def search(self, query: str, document: Optional[str] = None, 
              directory: Optional[str] = None, k: Optional[int] = None) -> Any:
        """
        🎯 MAIN SEARCH FUNCTION
        
        Args:
            query: Search query
            document: Single document to search in
            directory: Directory to search in
            k: Number of results (default: 5)
        
        Examples:
            pipeline.search("What is radar testing?")
            pipeline.search("system requirements", document="SRS.pdf")
            pipeline.search("test procedures", directory="/path/to/docs")
        """
        engine = self.initialize_query_engine()
        return engine.quick_search(query, document, directory, k)
    
    def search_multiple_docs(self, query: str, *documents: str, k: Optional[int] = None) -> Any:
        """
        Search across multiple specific documents.
        
        Example:
            pipeline.search_multiple_docs("performance", "doc1.pdf", "doc2.docx")
        """
        engine = self.initialize_query_engine()
        return engine.search_multiple_documents(query, *documents, k=k)
    
    def show_available_documents(self) -> None:
        """Show all available documents in the database."""
        engine = self.initialize_query_engine()
        engine.show_database_info()
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def test_filter(self, document: Optional[str] = None, 
                   directory: Optional[str] = None) -> Dict[str, Any]:
        """Test document filtering for debugging."""
        engine = self.initialize_query_engine()
        return engine.test_filter(document, directory)
    
    def validate_setup(self) -> bool:
        """Validate that the system is properly set up."""
        print("🔍 Validating RAG Pipeline Setup...")
        
        # Check if data directory exists
        if not os.path.exists(self.config.DATA_PATH):
            print(f"❌ Data directory not found: {self.config.DATA_PATH}")
            return False
        print(f"✅ Data directory exists: {self.config.DATA_PATH}")
        
        # Check if there are supported files
        supported_files = []
        for root, dirs, files in os.walk(self.config.DATA_PATH):
            for file in files:
                if any(file.lower().endswith(ext) for ext in self.config.SUPPORTED_EXTENSIONS):
                    supported_files.append(file)
        
        if not supported_files:
            print(f"❌ No supported files found in: {self.config.DATA_PATH}")
            return False
        print(f"✅ Found {len(supported_files)} supported files")
        
        # Check if database exists
        if os.path.exists(self.config.CHROMA_PATH):
            print(f"✅ Vector database exists: {self.config.CHROMA_PATH}")
        else:
            print(f"⚠️ Vector database not found: {self.config.CHROMA_PATH}")
            print("   Run pipeline.build_database() to create it")
        
        print("✅ Setup validation complete")
        return True
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._ollama_process:
            self._ollama_process.terminate()
            print("🧹 Ollama server stopped")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR EASY USAGE
# =============================================================================

# Global pipeline instance for convenience
_global_pipeline = None

def get_pipeline() -> RAGPipeline:
    """Get or create global pipeline instance."""
    global _global_pipeline
    if _global_pipeline is None:
        _global_pipeline = RAGPipeline()
    return _global_pipeline

def setup_rag_system() -> bool:
    """Setup the complete RAG system."""
    pipeline = get_pipeline()
    return pipeline.setup_system()

def build_database(force_rebuild: bool = False) -> None:
    """Build or update the document database."""
    pipeline = get_pipeline()
    pipeline.build_database(force_rebuild)

def search(query: str, document: str = None, directory: str = None, k: int = 5) -> Any:
    """
    Quick search function.
    
    Examples:
        search("What is radar testing?")
        search("system requirements", document="SRS.pdf")
        search("test procedures", directory="/path/to/docs")
    """
    pipeline = get_pipeline()
    return pipeline.search(query, document, directory, k)

def search_multiple_docs(query: str, *documents: str, k: int = 5) -> Any:
    """Search in multiple documents."""
    pipeline = get_pipeline()
    return pipeline.search_multiple_docs(query, *documents, k=k)

def show_documents() -> None:
    """Show available documents."""
    pipeline = get_pipeline()
    pipeline.show_available_documents()

def validate_setup() -> bool:
    """Validate system setup."""
    pipeline = get_pipeline()
    return pipeline.validate_setup()


# =============================================================================
# EXAMPLE USAGE AND MAIN EXECUTION
# =============================================================================

def main_example():
    """Example of how to use the RAG pipeline."""
    print("🚀 RAG Pipeline Example Usage")
    print("=" * 50)
    
    # Option 1: Use convenience functions (recommended)
    print("\n📖 Option 1: Using convenience functions")
    
    # Setup system (only needed once)
    if setup_rag_system():
        print("✅ System ready!")
    
    # Validate setup
    validate_setup()
    
    # Build/update database
    build_database()  # Uses smart update by default
    
    # Show available documents
    show_documents()
    
    # Perform searches
    search("What are the system requirements?")
    search("radar testing procedures", document="test_doc.pdf")
    search_multiple_docs("performance metrics", "doc1.pdf", "doc2.docx")
    
    print("\n" + "=" * 50)
    print("📖 Option 2: Using pipeline object")
    
    # Option 2: Use pipeline object directly
    pipeline = RAGPipeline()
    pipeline.setup_system()
    pipeline.build_database()
    pipeline.search("What is the project scope?")


if __name__ == "__main__":
    print("""
🚀 Professional RAG Pipeline v2.0.0
====================================

Quick Start:
1. setup_rag_system()           # Setup Ollama and models
2. build_database()             # Process documents
3. search("your question")      # Start searching!

Advanced Usage:
- search("query", document="file.pdf")
- search("query", directory="/path/to/docs")  
- search_multiple_docs("query", "doc1.pdf", "doc2.pdf")
- show_documents()              # See available files
- validate_setup()              # Check configuration

Configuration:
Edit RAGPipelineConfig class to customize:
- Document paths
- Model selection
- Processing parameters
""")
    
    # Run example if script is executed directly
    # main_example()  # Uncomment to run example