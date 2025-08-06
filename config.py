"""
Professional RAG Pipeline - Centralized Configuration
====================================================

Centralized configuration management for the entire RAG pipeline.
All settings should be managed from this single file.

Author: Mustafa Said Oğuztürk
Date: 2025-08-01
Version: 2.1.0 - Centralized Configuration
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
    """
    🎯 SINGLE SOURCE OF TRUTH - All configuration in one place
    
    Modify these values to change behavior across the entire pipeline.
    """
    
    # =============================================================================
    # 🗂️ CORE PATHS - Change these to match your setup
    # =============================================================================
    DATA_PATH = r"C:\Users\stajyer1\Desktop\Mustafa_Said_Oguzturk\data"  # Your document directory
    CHROMA_PATH = "chroma"  # Vector database storage path
    
    # =============================================================================
    # 🤖 MODEL CONFIGURATIONS
    # =============================================================================
    
    # Embedding Model Options (choose one by changing CURRENT_EMBEDDING_MODEL):
    EMBEDDING_MODELS = {
        "multilingual": "intfloat/multilingual-e5-large",          # Best for multilingual (Turkish + English)
        "english": "sentence-transformers/all-MiniLM-L6-v2",       # Faster, English-focused
        "turkish": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",  # Turkish-specific
        "performance": "SMARTICT/multilingual-e5-large-wiki-tr-rag"     # Optimized for Turkish RAG
    }
    CURRENT_EMBEDDING_MODEL = EMBEDDING_MODELS["multilingual"]  # 👈 Change this to switch models
    EMBEDDING_DEVICE = 'cpu'  # Use 'cpu' if no GPU | Use 'cuda' to accelerate
    
    # LLM Model Options:
    LLM_MODELS = {
        "llama3.2": {"model": "llama3.2", "temp": 0.1},                               # Default, good balance
        "qwen3": {"model": "qwen3:0.6b", "temp": 0.1},                                     # Alternative
        "turkish_mistral": {"model": "brooqs/mistral-turkish-v2:latest", "temp": 0.1},               # Fallback option
    }
    CURRENT_LLM = LLM_MODELS["llama3.2"]  # 👈 Change this to switch LLM models
    
    # =============================================================================
    # ⚙️ PROCESSING PARAMETERS
    # =============================================================================
    
    # Text Chunking
    CHUNK_SIZE = 800        # Characters per chunk      | Alternative: 1000
    CHUNK_OVERLAP = 400     # Overlap between chunks    | Alternative: 200
    
    # Search Parameters
    DEFAULT_K = 8           # Number of results to retrieve     | Alternative: 5
    MAX_TOC_RESULTS = 1     # Maximum Table of Contents results to keep
    
    # File Processing
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.doc', '.md', '.html', '.htm', '.rtf', '.odt', '.xlsx', '.xls'}
    
    # Table of Contents filtering patterns
    TOC_PATTERNS = [
        'table of contents', 'contents', 'içindekiler', 'İÇİNDEKİLER',
        'tablolar', 'şekiller', 'TABLOLAR', 'ŞEKİLLER'
    ]
    
    # =============================================================================
    # 🎛️ FEATURE FLAGS
    # =============================================================================
    ENABLE_TOC_FILTERING = True       # Filter Table of Contents automatically
    ENABLE_OCR = False                # OCR for scanned documents (slower)
    ENABLE_TABLE_EXTRACTION = True   # Extract tables from documents
    ENABLE_COMPREHENSIVE_LOGGING = False  # Detailed processing logs
    
    # =============================================================================
    # 📝 PROMPT TEMPLATE
    # =============================================================================
    PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
- -
Answer the question based on the above context: {question}
"""

    # =============================================================================
    # 🔧 UTILITY METHODS
    # =============================================================================
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """Get all configuration as a dictionary for easy access."""
        return {
            # Paths
            'data_path': cls.DATA_PATH,
            'chroma_path': cls.CHROMA_PATH,
            
            # Models
            'embedding_model': cls.CURRENT_EMBEDDING_MODEL,
            'embedding_device': cls.EMBEDDING_DEVICE,
            'llm_model': cls.CURRENT_LLM['model'],
            'llm_temperature': cls.CURRENT_LLM['temp'],
            
            # Processing
            'chunk_size': cls.CHUNK_SIZE,
            'chunk_overlap': cls.CHUNK_OVERLAP,
            'default_k': cls.DEFAULT_K,
            'max_toc_results': cls.MAX_TOC_RESULTS,
            'supported_extensions': cls.SUPPORTED_EXTENSIONS,
            'toc_patterns': cls.TOC_PATTERNS,
            
            # Features
            'enable_toc_filtering': cls.ENABLE_TOC_FILTERING,
            'enable_ocr': cls.ENABLE_OCR,
            'enable_table_extraction': cls.ENABLE_TABLE_EXTRACTION,
            'enable_comprehensive_logging': cls.ENABLE_COMPREHENSIVE_LOGGING,
            
            # Prompt
            'prompt_template': cls.PROMPT_TEMPLATE
        }
    
    @classmethod
    def print_config(cls):
        """Print current configuration for verification."""
        config = cls.get_config_dict()
        print("🔧 Current RAG Pipeline Configuration:")
        print("=" * 60)
        
        print("📁 Paths:")
        print(f"  Data Path: {config['data_path']}")
        print(f"  Chroma Path: {config['chroma_path']}")
        
        print("\n🤖 Models:")
        print(f"  Embedding Model: {config['embedding_model']}")
        print(f"  Embedding Device: {config['embedding_device']}")
        print(f"  LLM Model: {config['llm_model']}")
        print(f"  LLM Temperature: {config['llm_temperature']}")
        
        print("\n⚙️ Processing:")
        print(f"  Chunk Size: {config['chunk_size']}")
        print(f"  Chunk Overlap: {config['chunk_overlap']}")
        print(f"  Default K: {config['default_k']}")
        print(f"  Max ToC Results: {config['max_toc_results']}")
        
        print("\n🎛️ Features:")
        print(f"  ToC Filtering: {config['enable_toc_filtering']}")
        print(f"  OCR: {config['enable_ocr']}")
        print(f"  Table Extraction: {config['enable_table_extraction']}")
        print(f"  Comprehensive Logging: {config['enable_comprehensive_logging']}")
        
        print("=" * 60)


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
    def pull_llm_model(model_name: str = None) -> bool:
        """Pull LLM model if not already available."""
        if model_name is None:
            model_name = RAGPipelineConfig.CURRENT_LLM["model"]
        
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
        if not SystemSetup.pull_llm_model():
            return False
        
        print("=" * 50)
        print("✅ System setup complete!")
        return True


class RAGPipeline:
    """Main RAG Pipeline orchestrator that uses centralized configuration."""
    
    def __init__(self, config: Optional[RAGPipelineConfig] = None):
        """Initialize RAG Pipeline with centralized configuration."""
        self.config = config or RAGPipelineConfig()
        self.document_processor = None
        self.query_engine = None
        self._ollama_process = None
    
    def setup_system(self) -> bool:
        """Setup the complete system."""
        return SystemSetup.setup_complete_system()
    
    def initialize_document_processor(self):
        """Initialize document processor with centralized config."""
        if self.document_processor is None:
            from document_processing import RAGDocumentProcessor
            self.document_processor = RAGDocumentProcessor(
                data_path=self.config.DATA_PATH,
                chroma_path=self.config.CHROMA_PATH,
                config=self.config  # Pass the entire config
            )
        return self.document_processor
    
    def initialize_query_engine(self):
        """Initialize query engine with centralized config."""
        if self.query_engine is None:
            from query_engine import RAGQueryEngine
            self.query_engine = RAGQueryEngine(
                chroma_path=self.config.CHROMA_PATH,
                embedding_model=self.config.CURRENT_EMBEDDING_MODEL,
                llm_model=self.config.CURRENT_LLM["model"],
                llm_temperature=self.config.CURRENT_LLM["temp"],
                config=self.config  # Pass the entire config
            )
        return self.query_engine
    
    # =========================================================================
    # DOCUMENT PROCESSING METHODS
    # =========================================================================
    
    def build_database(self, force_rebuild: bool = False) -> None:
        """Build or rebuild the document database."""
        processor = self.initialize_document_processor()
        
        if force_rebuild:
            print("⚠️ REBUILDING DATABASE FROM SCRATCH")
            processor.build_initial_database()
        else:
            print("🔄 UPDATING DATABASE")
            processor.update_database()
    
    def update_database(self) -> Dict[str, int]:
        """Update database with new/modified files."""
        processor = self.initialize_document_processor()
        return processor.update_database()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        engine = self.initialize_query_engine()
        return engine.database_analyzer.get_database_info()
    
    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    
    def search(self, query: str, document: Optional[str] = None, 
              directory: Optional[str] = None, k: Optional[int] = None) -> Any:
        """🎯 MAIN SEARCH FUNCTION"""
        engine = self.initialize_query_engine()
        return engine.quick_search(query, document, directory, k)
    
    def search_multiple_docs(self, query: str, *documents: str, k: Optional[int] = None) -> Any:
        """Search across multiple specific documents."""
        engine = self.initialize_query_engine()
        return engine.search_multiple_documents(query, *documents, k=k)
    
    def show_available_documents(self) -> None:
        """Show all available documents in the database."""
        engine = self.initialize_query_engine()
        engine.show_database_info()
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def print_config(self) -> None:
        """Print current configuration."""
        self.config.print_config()
    
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

def search(query: str, document: str = None, directory: str = None, k: int = None) -> Any:
    """Quick search function using centralized config."""
    pipeline = get_pipeline()
    return pipeline.search(query, document, directory, k)

def search_multiple_docs(query: str, *documents: str, k: int = None) -> Any:
    """Search in multiple documents."""
    pipeline = get_pipeline()
    return pipeline.search_multiple_docs(query, *documents, k=k)

def show_documents() -> None:
    """Show available documents."""
    pipeline = get_pipeline()
    pipeline.show_available_documents()

def show_config() -> None:
    """Show current configuration."""
    pipeline = get_pipeline()
    pipeline.print_config()

def validate_setup() -> bool:
    """Validate system setup."""
    pipeline = get_pipeline()
    return pipeline.validate_setup()


# =============================================================================
# EXAMPLE USAGE AND MAIN EXECUTION
# =============================================================================

def main_example():
    """Example of how to use the RAG pipeline with centralized config."""
    print("🚀 RAG Pipeline Example Usage (Centralized Config)")
    print("=" * 60)
    
    # Show current configuration
    show_config()
    
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


if __name__ == "__main__":
    print("""
🚀 Professional RAG Pipeline v2.1.0 - Centralized Configuration
===============================================================

🎯 ALL CONFIGURATION IS NOW CENTRALIZED IN THIS FILE!

Quick Start:
1. Edit RAGPipelineConfig class above to set your paths and preferences
2. setup_rag_system()           # Setup Ollama and models
3. build_database()             # Process documents  
4. search("your question")      # Start searching!

Configuration Functions:
- show_config()                 # View current settings
- validate_setup()              # Check if everything is set up correctly

Search Functions:
- search("query", document="file.pdf")
- search("query", directory="/path/to/docs")  
- search_multiple_docs("query", "doc1.pdf", "doc2.pdf")
- show_documents()              # See available files

🔧 To change any setting, simply modify the RAGPipelineConfig class above!
""")
    
    # Run example if script is executed directly
    # main_example()  # Uncomment to run example