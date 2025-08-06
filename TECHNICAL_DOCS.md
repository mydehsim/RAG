# RAG Pipeline - Technical Documentation for Engineers

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture & Design Decisions](#architecture--design-decisions)
3. [Component Breakdown](#component-breakdown)
4. [Data Flow Analysis](#data-flow-analysis)
5. [Configuration System Explained](#configuration-system-explained)
6. [Code Structure & Key Functions](#code-structure--key-functions)
7. [Understanding Each Module](#understanding-each-module)

---

## System Overview

### What This System Does
The RAG (Retrieval-Augmented Generation) Pipeline is a document search and question-answering system that I built to handle multilingual documents (Turkish and English). It processes documents, converts them into searchable chunks, stores them in a vector database, and uses LLMs to generate contextual answers.

### Why I Built It This Way
I needed a system that could:
- Handle Turkish technical documents alongside English ones
- Work completely offline with local LLMs (Ollama)
- Update incrementally without reprocessing everything
- Provide a user-friendly web interface
- Be easily configurable without diving into code

### The Core Architecture I Implemented
```
Web UI (Chainlit) 
    ↓
Query Engine (search & response generation)
    ↓
ChromaDB (vector similarity search)
    ↓
Document Processor (chunking & embedding)
    ↓
Ollama (local LLM for responses)
```

---

## Architecture & Design Decisions

### Why Centralized Configuration (config.py)

**The Problem I Faced**: In earlier versions, configuration was scattered across multiple files. When I wanted to change the embedding model, I had to edit 3 different files.

**My Solution**: Created `RAGPipelineConfig` class as the single source of truth. Now everything is in one place.

```python
class RAGPipelineConfig:
    # All settings in one place
    DATA_PATH = r"C:\your\documents"
    CURRENT_EMBEDDING_MODEL = EMBEDDING_MODELS["multilingual"]
    CHUNK_SIZE = 800
    # ... etc
```

**Why This Matters**: 
- New engineers only need to look at one file to understand all settings
- Reduces bugs from mismatched configurations
- Makes A/B testing models trivial

### Why I Chose ChromaDB

**Options I Considered**:
- Pinecone: Cloud-based, not suitable for offline requirement
- Weaviate: Too complex for this use case
- FAISS: No built-in persistence
- **ChromaDB**: Local, persistent, simple API, good performance

**The Implementation**:
```python
self.db = Chroma(
    persist_directory=self.chroma_path,
    embedding_function=self.embeddings,
)
```

### Why Ollama for LLMs

**My Requirements**:
- Must work offline
- Need to support Turkish
- Should be easy to install

**Why Ollama Won**:
- Simple installation (one command)
- Good model selection
- Easy API
- Runs locally

---

## Component Breakdown

### 1. config.py - The Control Center

**What It Does**: Controls everything in the system.

**Key Classes I Created**:

```python
class RAGPipelineConfig:
    """All configuration in one place"""
    
class SystemSetup:
    """Handles Ollama installation and setup"""
    
class RAGPipeline:
    """Main orchestrator that uses the config"""
```

**Important Design Choice**: I made convenience functions at module level so users don't need to instantiate classes:

```python
# Users can just do:
search("my query")
# Instead of:
pipeline = RAGPipeline()
pipeline.search("my query")
```

### 2. document_processing.py - The Document Handler

**What It Does**: Processes all documents and manages the database.

**The Processing Pipeline I Built**:

1. **DocumentConverter**: Tries Docling first (advanced), falls back to basic loading
2. **ChunkProcessor**: Splits text with overlap to preserve context
3. **TableOfContentsFilter**: Removes redundant ToC entries (important for Turkish docs)
4. **FileManager**: Tracks file changes using MD5 hashes
5. **DatabaseManager**: Handles all ChromaDB operations

**Why Hash-Based Change Detection**:
```python
def calculate_file_hash(file_path: str) -> str:
    """I use MD5 to detect if a file changed"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()
```

This prevents reprocessing unchanged files, saving significant time.

### 3. query_engine.py - The Search Engine

**Version 2.2.0 Changes I Made**:

**Old Approach (v2.1.0)**:
```python
response = engine.search_documents(query)  # Just returned response
```

**New Approach (v2.2.0)**:
```python
sources, response = engine.search_documents(query)  # Returns both!
```

**Why This Change**: The UI needs to display sources separately with clickable links.

**Components I Built**:

1. **RAGQueryEngine**: Main orchestrator
2. **DatabaseAnalyzer**: Provides database statistics
3. **ChromaDBFilterBuilder**: Creates filters for document/directory search
4. **ResponseGenerator**: Interfaces with Ollama
5. **TableOfContentsFilter**: Fast ToC filtering for search results

### 4. app.py - The Web Interface

**What It Does**: Provides Chainlit-based web UI.

**Key Features I Implemented**:

1. **Settings Panel**: Live configuration updates
2. **Source Hyperlinking**: Makes file paths clickable
3. **Database Management**: Update/rebuild from UI
4. **Typing Animation**: Better UX (though simplified for stability)

**Why Global Variables**:
```python
config = None  # Global config
pipeline = None  # Global pipeline
```

I used globals because Chainlit's async handlers need persistent state across requests.

---

## Data Flow Analysis

### Document Processing Flow

Here's exactly what happens when you add documents:

```
1. You place files in DATA_PATH directory

2. System scans directory:
   - Filters by SUPPORTED_EXTENSIONS
   - Calculates MD5 hash for each file

3. Compares with database:
   - New file → Process
   - Changed file (different hash) → Delete old, process new
   - Same file (same hash) → Skip
   - Deleted file → Remove from database

4. For each file to process:
   - Load with DocumentConverter (tries Docling, falls back to basic)
   - Split into chunks (800 chars with 400 overlap by default)
   - Filter Table of Contents (if enabled)
   - Generate embeddings using HuggingFace model
   - Store in ChromaDB with metadata

5. Each chunk gets unique ID:
   Format: {filepath}:pg.{page}:{chunk_index}
   Example: /docs/report.pdf:pg.5:2
```

### Search Flow

Here's what happens when you search:

```
1. Query arrives from UI or API

2. Filter construction:
   - If document specified → filter by document
   - If directory specified → filter by directory
   - If both → intersection
   - If neither → search everything

3. ChromaDB search:
   - Convert query to embedding
   - Find k nearest neighbors (default k=8)
   - Apply filters if any

4. Post-processing:
   - Filter Table of Contents results (if enabled)
   - Keep max 1 ToC result (configurable)

5. Response generation:
   - Concatenate chunk contents as context
   - Send to Ollama with prompt template
   - Get response

6. Return to UI:
   - Sources list (for hyperlinks)
   - LLM response (for display)
```

---

## Configuration System Explained

### The Configuration Hierarchy I Created

```
RAGPipelineConfig
├── Paths
│   ├── DATA_PATH: Where your documents are
│   └── CHROMA_PATH: Where database is stored
│
├── Models
│   ├── EMBEDDING_MODELS: Available embedding models
│   ├── CURRENT_EMBEDDING_MODEL: Active embedding model
│   ├── LLM_MODELS: Available LLM models
│   └── CURRENT_LLM: Active LLM model
│
├── Processing
│   ├── CHUNK_SIZE: How big each chunk is
│   ├── CHUNK_OVERLAP: Overlap between chunks
│   └── DEFAULT_K: How many search results
│
└── Features
    ├── ENABLE_TOC_FILTERING: Filter table of contents
    ├── ENABLE_OCR: Process scanned documents
    └── ENABLE_TABLE_EXTRACTION: Extract tables
```

### Why These Specific Models

**Embedding Models I Included**:
- `multilingual`: Best for Turkish+English (my primary use case)
- `english`: Faster when only English docs
- `turkish`: Optimized for Turkish-only
- `performance`: Balanced option for Turkish RAG

**LLM Models I Configured**:
- `llama3.2`: Good general performance
- `qwen3`: Lightweight alternative
- `turkish_mistral`: For Turkish-heavy content

---

## Code Structure & Key Functions

### Entry Points

These are the main functions users interact with:

```python
# From config.py - convenience functions
setup_rag_system()          # One-time setup
build_database()            # Build/update database
search(query)               # Search documents
show_documents()            # List available docs
validate_setup()            # Check system health
```

### The RAGPipeline Class

This is the main orchestrator I built:

```python
class RAGPipeline:
    def __init__(self, config=None):
        # Uses centralized config
        self.config = config or RAGPipelineConfig()
        
    def search(self, query, document=None, directory=None, k=None):
        # Main search function with filters
        
    def build_database(self, force_rebuild=False):
        # Smart or full rebuild
        
    def get_database_stats(self):
        # Returns database info
```

### The Search Function Flow

Here's the actual code path for a search:

```python
# 1. User calls:
search("my query", document="file.pdf")

# 2. Goes to config.py global function:
def search(query, document=None, ...):
    pipeline = get_pipeline()  # Get or create global
    return pipeline.search(query, document, ...)

# 3. RAGPipeline.search calls:
def search(self, query, ...):
    engine = self.initialize_query_engine()
    return engine.quick_search(query, ...)

# 4. RAGQueryEngine.quick_search:
def quick_search(self, query, ...):
    sources, response = self.search_documents(query, ...)
    self.response_generator.format_response_output(response, sources)
    return response

# 5. The actual search in search_documents:
def search_documents(self, query, ...):
    # Build filter
    where_filter = self.filter_builder.build_filter(...)
    
    # Search database
    results = db.similarity_search_with_relevance_scores(query, k, filter=where_filter)
    
    # Filter ToC if enabled
    if enable_toc_filter:
        results = self.toc_filter.filter_toc_fast(results)
    
    # Generate response
    context = "\n".join([doc.page_content for doc, _ in results])
    response = self.response_generator.generate_response(context, query)
    
    # Extract sources
    sources = [doc.metadata.get("id") for doc, _ in results]
    
    return sources, response
```

---

## Understanding Each Module

### document_processing.py Details

**Why I Made These Classes**:

1. **DocumentConverter**: Handles multiple file formats with fallback
   - Tries advanced Docling first (better PDF handling)
   - Falls back to basic text extraction
   - This ensures something always works

2. **ChunkProcessor**: Intelligent text splitting
   - 800 char chunks with 400 overlap (50%)
   - Why overlap? Prevents losing context at boundaries
   - Example: "The radar system requires..." might be split badly without overlap

3. **TableOfContentsFilter**: Removes ToC noise
   - Turkish documents often have extensive ToCs
   - These pollute search results
   - Pattern matching for "İÇİNDEKİLER", dot patterns, page numbers

4. **FileManager**: Efficient file tracking
   - MD5 hashing detects changes
   - Prevents reprocessing unchanged files
   - Critical for large document sets

5. **DatabaseManager**: ChromaDB interface
   - Handles all vector database operations
   - Manages metadata filtering
   - Ensures data consistency

### query_engine.py Details

**Why I Structured It This Way**:

1. **RAGQueryEngine**: Central coordinator
   - Initializes all components
   - Orchestrates search flow
   - Handles both legacy and new return formats

2. **DatabaseAnalyzer**: Database insights
   - Shows what documents are available
   - Provides statistics
   - Helps with debugging

3. **ChromaDBFilterBuilder**: Smart filtering
   - Converts user-friendly filters to ChromaDB format
   - Handles document name matching
   - Manages directory filtering
   - Creates intersection filters

4. **ResponseGenerator**: LLM interface
   - Manages Ollama connection
   - Formats prompts consistently
   - The prompt template constrains responses to context

5. **TableOfContentsFilter**: Fast result filtering
   - Different from document processor version
   - Optimized for speed over accuracy
   - Runs on search results, not raw documents

### app.py Details

**The Chainlit Integration**:

```python
@cl.on_chat_start
# Initializes session, shows welcome

@cl.on_message  
# Handles search queries

@cl.on_settings_update
# Processes configuration changes
```

**Why I Simplified Animations**:
- Original typing animation caused stability issues
- Simplified to word-batch updates
- Better reliability > fancy effects

**Source Hyperlinking Implementation**:
```python
file_url = f"file:///{normalized_path}"
markdown = f"[{display_name}]({file_url})"
```
This makes sources clickable in the UI.

### demo.py Purpose

**Why I Created This**:
- Interactive walkthrough for new users
- Tests all components systematically
- Shows best practices
- Validates installation

**The Demo Flow**:
1. Shows centralized configuration (new in v2.1.0)
2. Sets up Ollama if needed
3. Validates system health
4. Builds database
5. Demonstrates various search types
6. Allows interactive exploration

---

## Key Algorithms & Logic

### Chunk ID Algorithm

```python
# Format: {source}:pg.{page}:{chunk_index}
# Example: /docs/report.pdf:pg.5:2

# Why this format?
# - source: Identifies file
# - page: Helps users locate content
# - chunk_index: Unique within page
```

### ToC Detection Logic

```python
def is_toc_content(self, content):
    # Check patterns from config
    for pattern in self.config.TOC_PATTERNS:
        if pattern in content.lower():
            return True
    
    # Check for dot patterns (........)
    if '........' in content:
        return True
    
    # Check for page number patterns
    # Lines ending with numbers after dots
```

### Filter Building Logic

```python
# The filter builder creates ChromaDB filters:

if document and directory:
    # Both specified - intersection
    filter = {"source": {"$in": matching_files}}
    
elif document:
    # Document only
    filter = {"source": {"$eq": file_path}}
    
elif directory:
    # Directory only  
    filter = {"source": {"$in": files_in_directory}}
    
else:
    # No filter - search everything
    filter = None
```

---

## Understanding the Codebase

### For New Engineers

**Start Here**:
1. Read `config.py` first - understand all settings
2. Run `demo.py` - see the system in action
3. Try modifying `CHUNK_SIZE` and see the effect
4. Add a print statement in `search_documents()` to trace flow

**Key Patterns to Understand**:

1. **Fallback Pattern**: Always have a backup
   ```python
   try:
       # Advanced method
   except:
       # Basic method that always works
   ```

2. **Configuration Pattern**: Everything comes from config
   ```python
   self.config = config or RAGPipelineConfig()
   ```

3. **Lazy Loading Pattern**: Don't initialize until needed
   ```python
   if self.db is None:
       self.db = Chroma(...)
   ```

**Common Modifications You Might Make**:

1. **Add a new file format**: Modify `DocumentConverter.load_document()`
2. **Change embedding model**: Edit `CURRENT_EMBEDDING_MODEL` in config
3. **Adjust search results**: Change `DEFAULT_K` in config
4. **Add new LLM**: Add to `LLM_MODELS` dictionary in config

### Debugging Tips

**If searches return nothing**:
1. Check `show_documents()` - are documents loaded?
2. Check `validate_setup()` - is system healthy?
3. Try without filters: `search("query")` with no document/directory
4. Check if ToC filtering is too aggressive

**If processing is slow**:
1. Disable OCR: `ENABLE_OCR = False`
2. Reduce chunk size: `CHUNK_SIZE = 500`
3. Use English model if no Turkish: `CURRENT_EMBEDDING_MODEL = EMBEDDING_MODELS["english"]`

**If Ollama fails**:
1. Run `setup_rag_system()` again
2. Check if Ollama is running: `curl http://localhost:11434`
3. Try different model: `CURRENT_LLM = LLM_MODELS["qwen3"]`

---

## Summary

This RAG Pipeline is a complete document search and Q&A system that I built with these priorities:

1. **Simplicity**: Centralized configuration, clear abstractions
2. **Reliability**: Fallback mechanisms, smart error handling
3. **Performance**: Incremental updates, efficient filtering
4. **Usability**: Web UI, clickable sources, live configuration

The system is designed to be extended. New engineers should:
- Start with configuration (`config.py`)
- Understand the data flow
- Follow existing patterns
- Test changes with `demo.py`

Every design decision was made to balance functionality with maintainability. The code prioritizes clarity over cleverness, making it easier for future developers to understand and modify.

---

**Author**: Mustafa Said Oğuztürk  
**Version**: 2.2.0  
**Last Updated**: 2025-08-01