# Professional RAG Pipeline v2.0.0

A comprehensive, production-ready Retrieval-Augmented Generation (RAG) pipeline with advanced document processing, smart filtering, and optimized search capabilities.

## 🚀 Features

### Document Processing
- **Multi-format Support**: PDF, DOCX, DOC, TXT, MD, HTML, XLSX, XLS
- **Advanced PDF Processing**: Table extraction, OCR support, page-aware chunking
- **Smart Chunking**: Overlapping chunks with metadata preservation
- **Table of Contents Filtering**: Automatic detection and filtering of ToC content
- **File Change Detection**: Incremental updates with hash-based change detection

### Search & Retrieval
- **Semantic Search**: High-quality multilingual embeddings
- **Advanced Filtering**: Document-specific and directory-based filtering
- **Smart Result Ranking**: ToC filtering and relevance optimization
- **Multi-document Search**: Search across multiple documents simultaneously

### LLM Integration
- **Ollama Integration**: Local LLM execution with multiple model support
- **Context-aware Responses**: Proper citation and source tracking
- **Configurable Models**: Easy switching between different LLM models

## 📁 Project Structure

```
rag_pipeline/
├── document_processor.py    # Document processing and database management
├── query_engine.py          # Search and response generation
├── config.py            # Main orchestration and configuration
└── README.md                    # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Minimum 8GB RAM
- 10GB+ free disk space

### Required Packages
```bash
pip install python-dotenv==1.0.1
pip install langchain==0.3.24
pip install langchain-community==0.3.14
pip install langchain-openai==0.1.8
pip install langchain-core==0.3.29
pip install langchain-ollama==0.2.2
pip install unstructured==0.14.4
pip install onnxruntime==1.17.1
pip install chromadb
pip install openai==1.31.1
pip install tiktoken==0.7.0
pip install pypdf==5.4.0
pip install pydantic==2.9.2
pip install langchain-docling
```

## 🚀 Quick Start

### 1. Basic Setup
```python
from config import setup_rag_system, build_database, search, show_documents

# Setup system (run once)
setup_rag_system()

# Build database from your documents
build_database()

# Start searching!
search("What are the main requirements?")
```

### 2. Configure Your Data Path
```python
from config import RAGPipelineConfig

# Edit the configuration
RAGPipelineConfig.DATA_PATH = "/path/to/your/documents"
RAGPipelineConfig.CHROMA_PATH = "your_vector_db"
```

### 3. Advanced Usage
```python
from config import RAGPipeline

# Create pipeline with custom config
pipeline = RAGPipeline()

# Search in specific document
pipeline.search("radar specifications", document="specs.pdf")

# Search in directory
pipeline.search("test procedures", directory="/docs/testing")

# Search multiple documents
pipeline.search_multiple_docs("performance metrics", "doc1.pdf", "doc2.pdf")

# Show available documents
pipeline.show_available_documents()
```

## 🔧 Configuration

### Model Configuration
```python
class RAGPipelineConfig:
    # Embedding Models
    EMBEDDING_MODELS = {
        "multilingual": "intfloat/multilingual-e5-large",          # Best for Turkish + English
        "english": "sentence-transformers/all-MiniLM-L6-v2",       # Faster, English-focused
        "turkish": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",  # Turkish-specific
        "performance": "SMARTICT/multilingual-e5-large-wiki-tr-rag"     # Optimized for Turkish RAG
    }
    
    # LLM Models
    LLM_MODELS = {
        "llama3.2": {"model": "llama3.2", "temp": 0.1},          
        "qwen3:0.6B": {"model": "qwen3", "temp": 0.1},         
        "nomic-embed-text": {"model": "nomic-embed-text", "temp": 0.1},               
    }
```

### Processing Parameters
```python
# Text Chunking
CHUNK_SIZE = 800        # Characters per chunk
CHUNK_OVERLAP = 400     # Overlap between chunks

# Search Parameters
DEFAULT_K = 8           # Number of results to retrieve
MAX_TOC_RESULTS = 1     # Maximum Table of Contents results

# Feature Flags
ENABLE_TOC_FILTERING = True      # Filter Table of Contents
ENABLE_OCR = False               # OCR for scanned documents
ENABLE_TABLE_EXTRACTION = True   # Extract tables from documents
```

## 📖 Usage Examples

### Basic Search
```python
# Search everything
search("What is the project scope?")

# Search with more results
search("technical specifications", k=10)
```

### Document-Specific Search
```python
# Search in one document
search("system requirements", document="SRS.pdf")

# Search in multiple documents
search_multiple_docs("test cases", "test_plan.pdf", "test_procedures.docx")
```

### Directory-Based Search
```python
# Search in specific directory
search("installation guide", directory="/docs/installation")

# Search in document within directory
search("API documentation", document="api.pdf", directory="/docs/technical")
```

### Database Management
```python
# Show what's in your database
show_documents()

# Update database with new files
build_database()  # Smart update (recommended)

# Force complete rebuild
build_database(force_rebuild=True)

# Get database statistics
pipeline = RAGPipeline()
stats = pipeline.get_database_stats()
print(f"Total chunks: {stats['total_chunks']}")
```

### Debugging and Testing
```python
# Validate system setup
validate_setup()

# Test document filtering
pipeline = RAGPipeline()
result = pipeline.test_filter(document="test.pdf", directory="/docs")
print(result)
```

## 🎯 Best Practices

### 1. Document Organization
```
documents/
├── specifications/
│   ├── requirements.pdf
│   └── technical_specs.docx
├── procedures/
│   ├── test_procedures.pdf
│   └── installation_guide.md
└── reports/
    ├── analysis_report.pdf
    └── performance_metrics.xlsx
```

### 2. Search Query Optimization
```python
# Good: Specific questions
search("What are the functional requirements for the radar system?")

# Good: Domain-specific terms
search("API endpoints authentication methods")

# Avoid: Too vague
search("information")  # Too broad

# Avoid: Single words
search("test")  # Too generic
```

### 3. Filter Usage
```python
# Use document filter for specific files
search("error codes", document="troubleshooting.pdf")

# Use directory filter for related documents
search("installation steps", directory="/docs/setup")

# Combine for precise targeting
search("API authentication", document="api.pdf", directory="/docs/technical")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. No Results Found
```python
# Check available documents
show_documents()

# Verify database has content
pipeline = RAGPipeline()
stats = pipeline.get_database_stats()
print(f"Database has {stats['total_chunks']} chunks")

# Test without filters
search("your query")  # No document/directory filters
```

#### 2. Ollama Connection Issues
```python
# Check if Ollama is running
setup_rag_system()  # Reinstalls and starts Ollama

# Verify model is downloaded
from rag_main_setup import SystemSetup
SystemSetup.pull_llm_model("llama3.2")
```

#### 3. GPU Memory Issues
```python
# Switch to CPU for embeddings
RAGPipelineConfig.EMBEDDING_DEVICE = 'cpu'

# Use smaller embedding model
RAGPipelineConfig.CURRENT_EMBEDDING_MODEL = RAGPipelineConfig.EMBEDDING_MODELS["english"]
```

#### 4. Database Corruption
```python
# Force rebuild database
build_database(force_rebuild=True)

# Or manually delete and rebuild
import shutil
shutil.rmtree("chroma")  # Delete database
build_database()  # Rebuild from scratch
```

### Performance Optimization

#### 1. Faster Processing
```python
# Disable OCR for faster processing
RAGPipelineConfig.ENABLE_OCR = False

# Use smaller chunks for faster search
RAGPipelineConfig.CHUNK_SIZE = 500
RAGPipelineConfig.CHUNK_OVERLAP = 200

# Use English-only model if documents are English
RAGPipelineConfig.CURRENT_EMBEDDING_MODEL = RAGPipelineConfig.EMBEDDING_MODELS["english"]
```

#### 2. Better Accuracy
```python
# Enable comprehensive logging for debugging
RAGPipelineConfig.ENABLE_COMPREHENSIVE_LOGGING = True

# Use larger context windows
RAGPipelineConfig.CHUNK_SIZE = 1200
RAGPipelineConfig.CHUNK_OVERLAP = 600

# Increase search results
search("your query", k=10)
```

## 🔄 Update Workflow

### Regular Maintenance
```python
# Daily/Weekly: Update database
build_database()  # Smart incremental update

# Check database health
validate_setup()

# Monitor database size
pipeline = RAGPipeline()
stats = pipeline.get_database_stats()
print(f"Database: {stats['total_chunks']} chunks, {len(stats['files'])} files")
```

### Adding New Documents
1. Place new documents in your `DATA_PATH`
2. Run `build_database()` - it will automatically detect and process new files
3. Old files that haven't changed won't be reprocessed

### Removing Documents
1. Delete documents from your `DATA_PATH`
2. Run `build_database()` - it will automatically remove deleted files from database

## 📊 Monitoring and Analytics

### Database Statistics
```python
pipeline = RAGPipeline()
stats = pipeline.get_database_stats()

print(f"Total documents: {len(stats['files'])}")
print(f"Total chunks: {stats['total_chunks']}")
print(f"Available directories: {len(stats['directories'])}")
```

### Search Performance
```python
import time

start_time = time.time()
result = search("your query")
end_time = time.time()

print(f"Search completed in {end_time - start_time:.2f} seconds")
```

## 🤝 Contributing

### Code Structure
- **rag_document_processor.py**: Document loading, processing, and database management
- **rag_query_engine.py**: Search functionality and response generation  
- **rag_main_setup.py**: Main orchestration and user-friendly interfaces

### Adding New Features
1. Follow the existing class-based structure
2. Add comprehensive docstrings
3. Include error handling
4. Update configuration classes as needed
5. Add examples to README

## 📝 License

MIT License - feel free to use and modify for your projects.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Verify your configuration matches the examples
3. Test with the provided example documents first
4. Check that all dependencies are properly installed

---

## 📈 Version History

### v2.0.0 (Current)
- Complete rewrite with professional architecture
- Advanced document processing with Docling
- Smart file change detection
- Improved search filtering
- Better error handling and logging
- Comprehensive configuration system

### v1.0.0 (Previous)
- Basic RAG functionality
- Simple document processing
- ChromaDB integration
- Basic search capabilities